"""MLflow tools. Uses MlflowClient only (no UI scraping)."""

from __future__ import annotations

import os
import statistics
from datetime import UTC, datetime

from mlflow.tracking import MlflowClient

from agent.config import get_settings
from agent.registry import ToolRejected, tool
from agent.schemas import (
    CompareExperimentsInput,
    CompareRunsInput,
    FindBestRunInput,
    ListRunsInput,
    ModelRegistryInput,
    NoInput,
    RunIdInput,
)

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

KEY_METRICS = ["f1", "precision", "recall", "accuracy", "roc_auc"]
MIN_SECONDARY = 0.85  # precision/recall floor used by the readiness check


def get_client() -> MlflowClient:
    return MlflowClient(tracking_uri=get_settings().mlflow_tracking_uri)


def _ts(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat() if ms else None


def _summary(run) -> dict:
    t = run.data.tags
    return {
        "run_id": run.info.run_id,
        "run_name": t.get("mlflow.runName"),
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": _ts(run.info.start_time),
        "metrics": {k: run.data.metrics[k] for k in KEY_METRICS if k in run.data.metrics},
        "data.version": t.get("data.version"),
        "quality_gate": t.get("quality_gate"),
    }


def _exp_ids(client: MlflowClient, experiment: str | None) -> list[str]:
    if experiment is None:
        return [e.experiment_id for e in client.search_experiments()]
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise ToolRejected(f"experiment '{experiment}' not found")
    return [exp.experiment_id]


@tool(
    "mlflow_list_experiments", "List MLflow experiments with run counts.", NoInput, module="mlflow"
)
def list_experiments(_: NoInput) -> list[dict]:
    c = get_client()
    out = []
    for e in c.search_experiments():
        runs = c.search_runs([e.experiment_id], max_results=1000)
        out.append(
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "lifecycle_stage": e.lifecycle_stage,
                "run_count": len(runs),
            }
        )
    return out


@tool(
    "mlflow_list_runs",
    "List recent runs (metrics, data version) for an experiment or all.",
    ListRunsInput,
    module="mlflow",
)
def list_runs(inp: ListRunsInput) -> list[dict]:
    c = get_client()
    runs = c.search_runs(
        _exp_ids(c, inp.experiment),
        filter_string=inp.filter or "",
        max_results=inp.max_results,
        order_by=[inp.order_by],
    )
    return [_summary(r) for r in runs]


@tool(
    "mlflow_get_run_details",
    "Params, metrics, provenance tags and artifacts of one run.",
    RunIdInput,
    module="mlflow",
)
def get_run_details(inp: RunIdInput) -> dict:
    c = get_client()
    run = c.get_run(inp.run_id)
    arts = [a.path for a in c.list_artifacts(inp.run_id)]
    tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.log-model")}
    return {
        **_summary(run),
        "params": dict(run.data.params),
        "all_metrics": dict(run.data.metrics),
        "tags": tags,
        "artifacts": arts,
    }


def _versions_by_run(c: MlflowClient) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rm in c.search_registered_models():
        aliases: dict[str, str] = dict(getattr(rm, "aliases", {}) or {})
        for mv in c.search_model_versions(f"name='{rm.name}'"):
            als = [a for a, v in aliases.items() if str(v) == str(mv.version)]
            out[mv.run_id] = {"model": rm.name, "version": mv.version, "aliases": als}
    return out


def _latest_data_version(runs) -> str | None:
    vs = [r.data.tags.get("data.version") for r in runs if r.data.tags.get("data.version")]
    vs = [v for v in vs if v != "untagged"]
    return max(vs) if vs else None


def readiness_checks(run, runs, versions: dict[str, dict]) -> dict:
    t, m = run.data.tags, run.data.metrics
    reg = versions.get(run.info.run_id)
    latest = _latest_data_version(runs)
    checks = {
        "quality_gate_passed": t.get("quality_gate") == "passed",
        "registered": reg is not None,
        "registry": reg,
        "has_alias": bool(reg and reg["aliases"]),
        "trained_on_latest_data_version": (t.get("data.version") == latest) if latest else None,
        "latest_data_version_seen": latest,
        "secondary_metrics_ok": all(m.get(k, 0) >= MIN_SECONDARY for k in ("precision", "recall")),
        "secondary_metric_floor": MIN_SECONDARY,
    }
    flags = [
        k
        for k in (
            "quality_gate_passed",
            "registered",
            "has_alias",
            "trained_on_latest_data_version",
            "secondary_metrics_ok",
        )
        if checks[k] is False
    ]
    checks["production_ready"] = not flags
    checks["failed_checks"] = flags
    return checks


@tool(
    "mlflow_find_best_run",
    "Find the best run by a metric (max/min) with runner-up, margin and production-readiness "
    "checks. Highest F1 alone does NOT mean production-ready.",
    FindBestRunInput,
    module="mlflow",
)
def find_best_run(inp: FindBestRunInput) -> dict:
    c = get_client()
    runs = c.search_runs(_exp_ids(c, inp.experiment), max_results=500)
    scored = [r for r in runs if inp.metric in r.data.metrics]
    if not scored:
        return {"best": None, "message": f"no runs with metric '{inp.metric}'"}
    # stable tie-break: newest run first
    scored.sort(
        key=lambda r: (r.data.metrics[inp.metric], r.info.start_time or 0),
        reverse=(inp.mode == "max"),
    )
    if inp.mode == "min":
        scored.sort(key=lambda r: (r.data.metrics[inp.metric], -(r.info.start_time or 0)))
    scored = scored[: inp.max_results]
    versions = _versions_by_run(c)
    best, second = scored[0], (scored[1] if len(scored) > 1 else None)
    ties = [
        r.info.run_id for r in scored if r.data.metrics[inp.metric] == best.data.metrics[inp.metric]
    ]
    return {
        "metric": inp.metric,
        "mode": inp.mode,
        "best": _summary(best),
        "runner_up": _summary(second) if second else None,
        "margin": (
            abs(best.data.metrics[inp.metric] - second.data.metrics[inp.metric]) if second else None
        ),
        "tied_run_ids": ties if len(ties) > 1 else [],
        "readiness_checks": readiness_checks(best, runs, versions),
    }


@tool(
    "mlflow_compare_runs",
    "Side-by-side comparison of runs (explicit ids or the last N) with per-metric best, "
    "parameter differences and readiness checks.",
    CompareRunsInput,
    module="mlflow",
)
def compare_runs(inp: CompareRunsInput) -> dict:
    c = get_client()
    all_runs = c.search_runs(
        _exp_ids(c, inp.experiment), max_results=500, order_by=["start_time DESC"]
    )
    runs = [c.get_run(i) for i in inp.run_ids] if inp.run_ids else all_runs[: inp.last_n]
    if not runs:
        return {"runs": [], "message": "no runs found"}
    versions = _versions_by_run(c)
    table = [
        {
            "run_id": r.info.run_id,
            "data.version": r.data.tags.get("data.version"),
            "quality_gate": r.data.tags.get("quality_gate"),
            **{m: r.data.metrics.get(m) for m in inp.metrics},
        }
        for r in runs
    ]
    best = {}
    for m in inp.metrics:
        vals = [(r.data.metrics[m], r.info.run_id) for r in runs if m in r.data.metrics]
        if vals:
            best[m] = max(vals)[1]
    keys = sorted({k for r in runs for k in r.data.params})
    param_diff = {
        k: {r.info.run_id: r.data.params.get(k) for r in runs}
        for k in keys
        if len({r.data.params.get(k) for r in runs}) > 1
    }
    return {
        "runs": table,
        "best_per_metric": best,
        "param_differences": param_diff,
        "readiness_checks": {r.info.run_id: readiness_checks(r, all_runs, versions) for r in runs},
    }


@tool(
    "mlflow_compare_experiments",
    "Compare two experiments on a metric (best, mean, std, count).",
    CompareExperimentsInput,
    module="mlflow",
)
def compare_experiments(inp: CompareExperimentsInput) -> dict:
    c = get_client()
    out = {}
    for label, name in (("a", inp.exp_a), ("b", inp.exp_b)):
        runs = c.search_runs(_exp_ids(c, name), max_results=500)
        vals = [r.data.metrics[inp.metric] for r in runs if inp.metric in r.data.metrics]
        out[label] = {
            "experiment": name,
            "run_count": len(runs),
            "runs_with_metric": len(vals),
            "best": max(vals) if vals else None,
            "mean": statistics.fmean(vals) if vals else None,
            "std": statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None),
        }
    return {"metric": inp.metric, **out}


@tool(
    "mlflow_get_model_registry",
    "Registered models, versions, aliases and source run ids.",
    ModelRegistryInput,
    module="mlflow",
)
def get_model_registry(inp: ModelRegistryInput) -> list[dict]:
    c = get_client()
    names = [rm.name for rm in c.search_registered_models()]
    if inp.model_name and inp.model_name not in names:
        raise ToolRejected(
            f"registered model '{inp.model_name}' not found; registered models: {names}. "
            "Omit model_name to list all (experiment names are not model names)."
        )
    out = []
    for rm in c.search_registered_models():
        if inp.model_name and rm.name != inp.model_name:
            continue
        aliases = dict(getattr(rm, "aliases", {}) or {})
        vers = [
            {
                "version": mv.version,
                "run_id": mv.run_id,
                "created": _ts(mv.creation_timestamp),
                "aliases": [a for a, v in aliases.items() if str(v) == str(mv.version)],
            }
            for mv in c.search_model_versions(f"name='{rm.name}'")
        ]
        out.append({"name": rm.name, "aliases": aliases, "versions": vers})
    return out
