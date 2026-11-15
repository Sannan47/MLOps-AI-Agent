"""Kubernetes tools (official client). No delete/exec/apply tools exist by design."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import wraps

from kubernetes.client.rest import ApiException

from agent.config import get_settings
from agent.registry import ToolRejected, tool
from agent.schemas import (
    DeploymentInput,
    EventsInput,
    K8sBase,
    PodInput,
    PodLogsInput,
    PodsInput,
    ScaleInput,
    ServiceInput,
)
from kubernetes import client as k8s
from kubernetes import config as k8s_config

REQ_TIMEOUT = 10
_loaded = {"done": False}


def get_apis() -> tuple[k8s.CoreV1Api, k8s.AppsV1Api]:
    if not _loaded["done"]:
        s = get_settings()
        try:
            k8s_config.load_kube_config(config_file=s.kubeconfig or None)
        except Exception:  # noqa: BLE001
            k8s_config.load_incluster_config()
        _loaded["done"] = True
    return k8s.CoreV1Api(), k8s.AppsV1Api()


def _check_ns(ns: str) -> str:
    if ns not in get_settings().allowed_namespaces:
        raise ToolRejected(f"namespace '{ns}' is not approved")
    return ns


def _rbac(fn):
    """Translate ApiException into clear errors (403 -> forbidden by RBAC)."""

    @wraps(fn)
    def wrapper(inp):
        _check_ns(getattr(inp, "namespace", get_settings().k8s_namespace))
        try:
            return fn(inp)
        except ApiException as e:
            if e.status == 403:
                raise RuntimeError(f"forbidden by RBAC (403): {e.reason}") from e
            if e.status == 404:
                raise RuntimeError(f"not found (404): {e.reason}") from e
            raise RuntimeError(f"Kubernetes API error {e.status}: {e.reason}") from e

    return wrapper


def _age_s(ts) -> float | None:
    return (datetime.now(UTC) - ts).total_seconds() if ts else None


def container_info(cs) -> dict:
    st = cs.state
    waiting = getattr(st, "waiting", None) if st else None
    term = getattr(st, "terminated", None) if st else None
    last = getattr(cs.last_state, "terminated", None) if cs.last_state else None
    return {
        "name": cs.name,
        "ready": cs.ready,
        "restart_count": cs.restart_count,
        "image": cs.image,
        "waiting_reason": waiting.reason if waiting else None,
        "terminated_reason": term.reason if term else None,
        "last_terminated_reason": last.reason if last else None,
        "last_exit_code": last.exit_code if last else None,
    }


def pod_summary(p) -> dict:
    statuses = [container_info(c) for c in (p.status.container_statuses or [])]
    ready = bool(statuses) and all(c["ready"] for c in statuses)
    return {
        "name": p.metadata.name,
        "phase": p.status.phase,
        "ready": ready,
        "restarts": sum(c["restart_count"] or 0 for c in statuses),
        "node": p.spec.node_name,
        "age_seconds": _age_s(p.metadata.creation_timestamp),
        "containers": statuses,
    }


def unhealthy_reasons(s: dict) -> list[str]:
    r = []
    if s["phase"] not in ("Running", "Succeeded") or not s["ready"]:
        r.append(f"not ready (phase={s['phase']})")
    if s["restarts"] > 0:
        r.append(f"{s['restarts']} restart(s)")
    for c in s["containers"]:
        for key in ("waiting_reason", "terminated_reason", "last_terminated_reason"):
            if c[key] and c[key] not in ("Completed",):
                r.append(f"{c['name']}: {c[key]}")
    return r


def _event_dict(e) -> dict:
    return {
        "type": e.type,
        "reason": e.reason,
        "message": (e.message or "")[:300],
        "object": f"{e.involved_object.kind}/{e.involved_object.name}",
        "count": e.count,
        "last_seen": (e.last_timestamp or e.event_time or e.metadata.creation_timestamp),
    }


def _events(ns: str, involved: str | None, warning_only: bool) -> list[dict]:
    core, _ = get_apis()
    items = core.list_namespaced_event(ns, _request_timeout=REQ_TIMEOUT).items
    out = [_event_dict(e) for e in items]
    if warning_only:
        out = [e for e in out if e["type"] == "Warning"]
    if involved:
        out = [e for e in out if involved in e["object"]]
    out.sort(key=lambda e: str(e["last_seen"]), reverse=True)
    for e in out:
        e["last_seen"] = str(e["last_seen"])
    return out[:50]


@tool(
    "k8s_list_deployments",
    "List deployments: replicas, images, version label, conditions.",
    K8sBase,
    module="k8s",
)
@_rbac
def list_deployments(inp: K8sBase) -> list[dict]:
    _, apps = get_apis()
    out = []
    for d in apps.list_namespaced_deployment(inp.namespace, _request_timeout=REQ_TIMEOUT).items:
        labels = d.metadata.labels or {}
        out.append(
            {
                "name": d.metadata.name,
                "desired": d.spec.replicas,
                "ready": d.status.ready_replicas or 0,
                "available": d.status.available_replicas or 0,
                "images": [c.image for c in d.spec.template.spec.containers],
                "version_label": labels.get("app.kubernetes.io/version"),
                "conditions": [
                    {"type": c.type, "status": c.status, "reason": c.reason}
                    for c in (d.status.conditions or [])
                ],
            }
        )
    return out


@tool(
    "k8s_list_pods",
    "List pods with phase, readiness, restarts and container states.",
    PodsInput,
    module="k8s",
)
@_rbac
def list_pods(inp: PodsInput) -> list[dict]:
    core, _ = get_apis()
    pods = core.list_namespaced_pod(
        inp.namespace, label_selector=inp.label_selector or "", _request_timeout=REQ_TIMEOUT
    ).items
    return [pod_summary(p) for p in pods]


@tool(
    "k8s_get_pod_status",
    "Detailed status of one pod, incl. last termination and probes.",
    PodInput,
    module="k8s",
)
@_rbac
def get_pod_status(inp: PodInput) -> dict:
    core, _ = get_apis()
    p = core.read_namespaced_pod(inp.pod, inp.namespace, _request_timeout=REQ_TIMEOUT)
    probes = [
        {
            "container": c.name,
            "readiness": getattr(getattr(c.readiness_probe, "http_get", None), "path", None),
            "liveness": getattr(getattr(c.liveness_probe, "http_get", None), "path", None),
        }
        for c in p.spec.containers
    ]
    return {**pod_summary(p), "probes": probes}


@tool(
    "k8s_get_pod_logs",
    "Tail of a pod's logs (previous=true for crashed containers).",
    PodLogsInput,
    module="k8s",
)
@_rbac
def get_pod_logs(inp: PodLogsInput) -> dict:
    core, _ = get_apis()
    text = core.read_namespaced_pod_log(
        inp.pod,
        inp.namespace,
        container=inp.container,
        tail_lines=inp.tail_lines,
        previous=inp.previous,
        _request_timeout=REQ_TIMEOUT,
    )
    return {"pod": inp.pod, "previous": inp.previous, "log_tail": str(text).splitlines()}


def _service_info(core, ns: str, svc) -> dict:
    ep = None
    try:
        ep = core.read_namespaced_endpoints(svc.metadata.name, ns, _request_timeout=REQ_TIMEOUT)
    except ApiException as e:
        if e.status != 404:
            raise
    ready = sum(len(s.addresses or []) for s in (ep.subsets or [])) if ep else 0
    not_ready = sum(len(s.not_ready_addresses or []) for s in (ep.subsets or [])) if ep else 0
    return {
        "name": svc.metadata.name,
        "type": svc.spec.type,
        "cluster_ip": svc.spec.cluster_ip,
        "ports": [
            {"port": p.port, "target_port": str(p.target_port)} for p in svc.spec.ports or []
        ],
        "selector": svc.spec.selector,
        "ready_endpoints": ready,
        "not_ready_endpoints": not_ready,
        "healthy_backends": ready > 0,
    }


@tool(
    "k8s_list_services",
    "List services with endpoint counts (0 ready = no healthy backends).",
    K8sBase,
    module="k8s",
)
@_rbac
def list_services(inp: K8sBase) -> list[dict]:
    core, _ = get_apis()
    svcs = core.list_namespaced_service(inp.namespace, _request_timeout=REQ_TIMEOUT).items
    return [_service_info(core, inp.namespace, s) for s in svcs]


@tool(
    "k8s_get_events",
    "Recent namespace events (warnings by default), newest first.",
    EventsInput,
    module="k8s",
)
@_rbac
def get_events(inp: EventsInput) -> list[dict]:
    return _events(inp.namespace, inp.involved_object, inp.warning_only)


@tool(
    "k8s_find_unhealthy_pods",
    "Find pods that are not ready, restarting, or in a bad state.",
    K8sBase,
    module="k8s",
)
@_rbac
def find_unhealthy_pods(inp: K8sBase) -> dict:
    core, _ = get_apis()
    pods = core.list_namespaced_pod(inp.namespace, _request_timeout=REQ_TIMEOUT).items
    bad = []
    for p in pods:
        s = pod_summary(p)
        reasons = unhealthy_reasons(s)
        if reasons:
            bad.append(
                {**s, "reasons": reasons, "events": _events(inp.namespace, s["name"], False)[:5]}
            )
    return {"unhealthy_pods": bad, "total_pods": len(pods)}


@tool(
    "k8s_get_deployed_version",
    "Deployed image tag, version label and ConfigMap APP_VERSION/MODEL_VERSION.",
    DeploymentInput,
    module="k8s",
)
@_rbac
def get_deployed_version(inp: DeploymentInput) -> dict:
    core, apps = get_apis()
    d = apps.read_namespaced_deployment(inp.deployment, inp.namespace, _request_timeout=REQ_TIMEOUT)
    images = [c.image for c in d.spec.template.spec.containers]
    cm_names = []
    for c in d.spec.template.spec.containers:
        for ef in c.env_from or []:
            if ef.config_map_ref:
                cm_names.append(ef.config_map_ref.name)
    cfg: dict = {}
    for n in cm_names:
        cm = core.read_namespaced_config_map(n, inp.namespace, _request_timeout=REQ_TIMEOUT)
        for k in ("APP_VERSION", "MODEL_VERSION", "MODEL_PATH"):
            if k in (cm.data or {}):
                cfg[k] = cm.data[k]
    return {
        "deployment": inp.deployment,
        "images": images,
        "image_tags": [
            i.rsplit(":", 1)[-1] if ":" in i.split("/")[-1] else "latest" for i in images
        ],
        "version_label": (d.metadata.labels or {}).get("app.kubernetes.io/version"),
        "configmap": cfg,
        "ready_replicas": d.status.ready_replicas or 0,
        "desired_replicas": d.spec.replicas,
    }


@tool(
    "k8s_diagnose_service",
    "Read-only diagnosis: service, endpoints, pods, events and logs with findings.",
    ServiceInput,
    module="k8s",
)
@_rbac
def diagnose_service(inp: ServiceInput) -> dict:
    core, _ = get_apis()
    ns = inp.namespace
    svc = core.read_namespaced_service(inp.service, ns, _request_timeout=REQ_TIMEOUT)
    info = _service_info(core, ns, svc)
    selector = ",".join(f"{k}={v}" for k, v in (svc.spec.selector or {}).items())
    pods = [
        pod_summary(p)
        for p in core.list_namespaced_pod(
            ns, label_selector=selector, _request_timeout=REQ_TIMEOUT
        ).items
    ]
    findings: list[str] = []
    if not info["healthy_backends"]:
        findings.append(f"Service '{inp.service}' has 0 ready endpoints (no healthy backends)")
    if not pods:
        findings.append("no pods match the service selector")
    logs: dict[str, list[str]] = {}
    for s in pods:
        reasons = unhealthy_reasons(s)
        if reasons:
            findings.append(f"pod {s['name']}: " + ", ".join(reasons))
            try:
                text = core.read_namespaced_pod_log(
                    s["name"], ns, tail_lines=20, _request_timeout=REQ_TIMEOUT
                )
                logs[s["name"]] = str(text).splitlines()
            except ApiException:
                pass  # container may not have started (e.g. ImagePullBackOff)
    events = _events(ns, None, True)
    for e in events[:5]:
        findings.append(f"event {e['reason']} on {e['object']}: {e['message']}")
    return {"service": info, "pods": pods, "events": events, "logs": logs, "findings": findings}


@tool(
    "k8s_restart_deployment",
    "Request a rolling restart of a deployment (rollout restart). Requires user confirmation.",
    DeploymentInput,
    risk="mutate",
    module="k8s",
)
@_rbac
def restart_deployment(inp: DeploymentInput) -> dict:
    _, apps = get_apis()
    now = datetime.now(UTC).isoformat()
    body = {
        "spec": {
            "template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}}
        }
    }
    apps.patch_namespaced_deployment(
        inp.deployment, inp.namespace, body, _request_timeout=REQ_TIMEOUT
    )
    return {"deployment": inp.deployment, "restarted_at": now}


@tool(
    "k8s_scale_deployment",
    "Request scaling a deployment to 1-5 replicas. Requires user confirmation.",
    ScaleInput,
    risk="mutate",
    module="k8s",
)
@_rbac
def scale_deployment(inp: ScaleInput) -> dict:
    _, apps = get_apis()
    apps.patch_namespaced_deployment_scale(
        inp.deployment,
        inp.namespace,
        {"spec": {"replicas": inp.replicas}},
        _request_timeout=REQ_TIMEOUT,
    )
    return {"deployment": inp.deployment, "replicas": inp.replicas}
