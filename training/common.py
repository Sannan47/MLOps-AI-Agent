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


def _tag_md5(tag: str) -> str:
    return _dvc_md5(_git("show", f"{tag}:{DATA_DVC}"))


def _data_provenance(md5: str, data_rev: str | None) -> tuple[str, str]:
    """Resolve (version tag, commit) for the dataset. Prefer an explicit revision, then a tag
    whose recorded md5 equals the data's md5, then fall back to git describe."""
    if data_rev:
        return data_rev, _git("rev-list", "-n1", data_rev)
    for tag in _git("tag", "-l", "data-v*").split():
        if md5 and _tag_md5(tag) == md5:
            return tag, _git("rev-list", "-n1", tag)
    return (
        _git("describe", "--tags", "--match", "data-v*", "--abbrev=0") or "untagged",
        _git("log", "-n1", "--format=%H", "--", DATA_DVC) or "unknown",
    )


def capture_provenance(
    trigger: str | None = None, dag_run_id: str | None = None, data_rev: str | None = None
) -> dict[str, str]:
    """Return the provenance tags recorded on every MLflow run.

    `data_rev` (or env DATA_REV) pins the dataset to a data-v* tag when the data was fetched
    for a specific revision rather than taken from the workspace."""
    data_rev = data_rev or os.environ.get("DATA_REV") or None
    dvc_file = REPO_ROOT / DATA_DVC
    if data_rev:
        md5 = _tag_md5(data_rev)
    else:
        md5 = _dvc_md5(dvc_file.read_text(encoding="utf-8")) if dvc_file.exists() else ""
    version, commit = _data_provenance(md5, data_rev)
    tags = {
        "data.path": "data/raw/dataset.csv",
        "data.dvc_md5": md5 or "unknown",
        "data.version": version or "untagged",
        "data.git_commit": commit or "unknown",
        "data.dvc_remote": _dvc_remote() or "unknown",
        "code.git_commit": _git("rev-parse", "HEAD") or "unknown",
        "code.git_dirty": "true" if _git("status", "--porcelain") else "false",
        "pipeline.trigger": trigger or os.environ.get("PIPELINE_TRIGGER", "manual"),
    }
    dag_run_id = dag_run_id or os.environ.get("AIRFLOW_DAG_RUN_ID")
    if dag_run_id:
        tags["airflow.dag_run_id"] = dag_run_id
    return tags
