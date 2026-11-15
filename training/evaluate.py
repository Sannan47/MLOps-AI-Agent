"""Evaluate on the test split; log metrics + confusion matrix to the same MLflow run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from training.common import load_params, tracking_uri  # noqa: E402


def compute_metrics(model, test: pd.DataFrame, label: str) -> tuple[dict, list]:
    x, y = test.drop(columns=[label]), test[label]
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]
    m = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }
    return {k: float(v) for k, v in m.items()}, confusion_matrix(y, pred).tolist()


def evaluate(
    processed: str | Path,
    model_path: str | Path,
    run_id: str,
    params: dict,
    metrics_out: str | Path = "metrics.json",
) -> dict:
    model = joblib.load(model_path)
    test = pd.read_csv(Path(processed) / "test.csv")
    metrics, cm = compute_metrics(model, test, params["data"]["label"])
    Path(metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    png = Path(metrics_out).with_name("confusion_matrix.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    mlflow.set_tracking_uri(tracking_uri())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(png))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--model", default="models/model.joblib")
    p.add_argument("--run-id-file", default="models/run_id.txt")
    p.add_argument("--metrics-out", default="metrics.json")
    p.add_argument("--params", default=None)
    a = p.parse_args()
    run_id = Path(a.run_id_file).read_text(encoding="utf-8").strip()
    print(evaluate(a.processed, a.model, run_id, load_params(a.params), a.metrics_out))


if __name__ == "__main__":
    main()
