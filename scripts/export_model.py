"""Download the `champion` model (+ scaler) from the MLflow registry into serving/model/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mlflow.tracking import MlflowClient

from training.common import load_params, tracking_uri


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="serving/model")
    p.add_argument("--alias", default="champion")
    a = p.parse_args()
    params = load_params()
    name = params["mlflow"]["model_name"]
    client = MlflowClient(tracking_uri())
    mv = client.get_model_version_by_alias(name, a.alias)
    run = client.get_run(mv.run_id)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_dl"
    model = client.download_artifacts(mv.run_id, "model_file/model.joblib", str(tmp))
    shutil.copy(model, out / "model.joblib")
    scaler = client.download_artifacts(mv.run_id, "scaler/scaler.joblib", str(tmp))
    shutil.copy(scaler, out / "scaler.joblib")
    shutil.rmtree(tmp, ignore_errors=True)
    meta = {
        "model_name": name,
        "model_version": str(mv.version),
        "run_id": mv.run_id,
        "data_version": run.data.tags.get("data.version"),
        "data_dvc_md5": run.data.tags.get("data.dvc_md5"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()
