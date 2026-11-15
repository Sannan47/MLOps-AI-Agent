"""Quality gate: min F1 and no regression vs champion. Promotion only on pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from training.common import QualityGateError, load_params, tracking_uri


def _champion_f1(client: MlflowClient, name: str) -> float | None:
    try:
        mv = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None
    return client.get_run(mv.run_id).data.metrics.get("f1")


def apply_gate(
    run_id: str, params: dict, min_f1: float | None = None, client: MlflowClient | None = None
) -> dict:
    client = client or MlflowClient(tracking_uri())
    name = params["mlflow"]["model_name"]
    g = params["gate"]
    threshold = g["min_f1"] if min_f1 is None else min_f1
    f1 = client.get_run(run_id).data.metrics.get("f1")
    if f1 is None:
        client.set_tag(run_id, "quality_gate", "failed")
        raise QualityGateError("QualityGateError: run has no f1 metric")
    champ = _champion_f1(client, name)
    reasons = []
    if f1 < threshold:
        reasons.append(f"f1 {f1:.4f} < min_f1 {threshold}")
    if champ is not None and f1 < champ - g["max_f1_drop_vs_champion"]:
        reasons.append(
            f"f1 {f1:.4f} regressed vs champion {champ:.4f} by more than "
            f"{g['max_f1_drop_vs_champion']}"
        )
    if reasons:
        client.set_tag(run_id, "quality_gate", "failed")
        client.set_tag(run_id, "quality_gate_reason", "; ".join(reasons))
        raise QualityGateError("QualityGateError: " + "; ".join(reasons))
    # pass branch: the only place a model is registered / aliased
    client.set_tag(run_id, "quality_gate", "passed")
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    alias = "champion" if champ is None else "challenger"
    client.set_registered_model_alias(name, alias, mv.version)
    client.set_tag(run_id, "registered_version", str(mv.version))
    return {"passed": True, "f1": f1, "version": mv.version, "alias": alias}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id-file", default="models/run_id.txt")
    p.add_argument("--min-f1", type=float, default=None)
    a = p.parse_args()
    mlflow.set_tracking_uri(tracking_uri())
    run_id = Path(a.run_id_file).read_text(encoding="utf-8").strip()
    try:
        print(apply_gate(run_id, load_params(), a.min_f1))
    except QualityGateError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
