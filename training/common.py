"""Shared helpers: params, errors, MLflow setup and dataset provenance capture."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DVC = "data/raw/dataset.csv.dvc"


class DataValidationError(Exception):
    """Raised when the dataset fails validation."""


class QualityGateError(Exception):
    """Raised when a model fails the quality gate."""


def load_params(path: str | Path | None = None) -> dict:
    p = Path(path) if path else REPO_ROOT / "params.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def tracking_uri() -> str:
    return (
        os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
    )


def _git(*args: str) -> str:
    try:
        r = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=REPO_ROOT, timeout=20, shell=False
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _dvc_md5(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("- ")
        if line.startswith("md5:"):
            return line.split(":", 1)[1].strip()
    return ""


def _dvc_remote() -> str:
    try:
        cfg = (REPO_ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    except OSError:
        return ""
    name = url = ""
    for line in cfg.splitlines():
        line = line.strip()
        if line.startswith("remote ="):
            name = line.split("=", 1)[1].strip()
        if line.startswith("url ="):
            url = line.split("=", 1)[1].strip()
    return f"{name} ({url})" if name else ""


def capture_provenance(trigger: str | None = None, dag_run_id: str | None = None) -> dict[str, str]:
    """Return the provenance tags recorded on every MLflow run."""
    dvc_file = REPO_ROOT / DATA_DVC
    md5 = _dvc_md5(dvc_file.read_text(encoding="utf-8")) if dvc_file.exists() else ""
    tags = {
        "data.path": "data/raw/dataset.csv",
        "data.dvc_md5": md5 or "unknown",
        "data.version": _git("describe", "--tags", "--match", "data-v*", "--abbrev=0")
        or "untagged",
        "data.git_commit": _git("log", "-n1", "--format=%H", "--", DATA_DVC) or "unknown",
        "data.dvc_remote": _dvc_remote() or "unknown",
        "code.git_commit": _git("rev-parse", "HEAD") or "unknown",
        "code.git_dirty": "true" if _git("status", "--porcelain") else "false",
        "pipeline.trigger": trigger or os.environ.get("PIPELINE_TRIGGER", "manual"),
    }
    dag_run_id = dag_run_id or os.environ.get("AIRFLOW_DAG_RUN_ID")
    if dag_run_id:
        tags["airflow.dag_run_id"] = dag_run_id
    return tags
