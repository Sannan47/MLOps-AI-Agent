"""Populate MLflow with realistic run history across dataset versions and hyperparameters.

Every run trains on the dataset fetched with `dvc get --rev <tag>` and records that tag as
provenance (env DATA_REV). Runs that fail the quality gate are kept (tagged failed)."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import yaml

from training import evaluate, preprocess, quality_gate, train, validate
from training.common import QualityGateError, load_params, tracking_uri

RUNS = [
    # (experiment, data tag, train params, gate min_f1 override)
    ("breast-cancer-classifier", "data-v1.0", {"model": "logistic_regression", "C": 1.0}, None),
    ("breast-cancer-classifier", "data-v1.1", {"model": "logistic_regression", "C": 1.0}, None),
    ("breast-cancer-classifier", "data-v1.1", {"model": "logistic_regression", "C": 10.0}, None),
    ("breast-cancer-classifier", "data-v2.0", {"model": "logistic_regression", "C": 1.0}, None),
    (
        "breast-cancer-classifier",
        "data-v2.0",
        {"model": "random_forest", "n_estimators": 200, "max_depth": 8},
        None,
    ),
    (
        "breast-cancer-classifier",
        "data-v2.0",
        {"model": "random_forest", "n_estimators": 1, "max_depth": 1},
        None,
    ),  # low capacity
    ("breast-cancer-baselines", "data-v1.0", {"model": "logistic_regression", "C": 0.01}, None),
    (
        "breast-cancer-baselines",
        "data-v1.1",
        {"model": "random_forest", "n_estimators": 10, "max_depth": 2},
        None,
    ),
    ("breast-cancer-baselines", "data-v2.0", {"model": "logistic_regression", "C": 0.1}, None),
]


def dvc_get(tag: str, dest: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dvc",
            "get",
            ".",
            "data/raw/dataset.csv",
            "--rev",
            tag,
            "-o",
            str(dest),
            "-f",
        ],
        check=True,
    )


def main() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    base = load_params()
    for exp, tag, overrides, min_f1 in RUNS:
        params = copy.deepcopy(base)
        params["train"].update(overrides)
        params["mlflow"]["experiment"] = exp
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            csv = t / "dataset.csv"
            dvc_get(tag, csv)
            os.environ["DATA_REV"] = tag
            pfile = t / "params.yaml"
            pfile.write_text(yaml.safe_dump(params), encoding="utf-8")
            report = t / "validation_report.json"
            validate.validate_file(csv, params, report)
            preprocess.preprocess(csv, t / "proc", params)
            name = f"{tag}-{params['train']['model']}"
            run_id = train.train(
                t / "proc",
                t / "model.joblib",
                params,
                run_name=name,
                validation_report=report,
                params_path=pfile,
            )
            m = evaluate.evaluate(t / "proc", t / "model.joblib", run_id, params, t / "m.json")
        gate = "n/a"
        # only the main experiment feeds the registry; baselines are compared, not promoted
        if exp == "breast-cancer-classifier":
            try:
                gate = quality_gate.apply_gate(run_id, params, min_f1)
            except QualityGateError as e:
                gate = f"FAILED: {e}"
        print(f"{exp} {name} f1={m['f1']:.4f} -> {gate}", flush=True)


if __name__ == "__main__":
    main()
