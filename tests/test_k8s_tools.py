from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from kubernetes.client.rest import ApiException
from urllib3.exceptions import MaxRetryError

from agent.registry import registry


def container(name="api", ready=True, restarts=0, waiting=None, last_term=None):
    return NS(
        name=name,
        ready=ready,
        restart_count=restarts,
        image="mlops-inference:1.0.0",
        state=NS(waiting=NS(reason=waiting) if waiting else None, terminated=None),
        last_state=NS(terminated=NS(reason=last_term, exit_code=137) if last_term else None),
    )


def pod(name, phase="Running", containers=None):
    return NS(
        metadata=NS(name=name, creation_timestamp=datetime.now(UTC)),
        status=NS(phase=phase, container_statuses=containers or [container()]),
        spec=NS(node_name="node1", containers=[]),
    )


def apis(pods=(), events=(), pods_exc=None):
    core, apps = MagicMock(), MagicMock()
    if pods_exc:
        core.list_namespaced_pod.side_effect = pods_exc
    else:
        core.list_namespaced_pod.return_value = NS(items=list(pods))
    core.list_namespaced_event.return_value = NS(items=list(events))
    return core, apps


def run(tool, args, core_apps):
    with patch("agent.tools.kubernetes_tools.get_apis", lambda: core_apps):
        return registry.execute(tool, args, "r")


def test_k8s_find_unhealthy_pods_detects_crashloop_and_imagepull():
    pods = [
        pod("good"),
        pod(
            "crash",
            containers=[
                container(
                    ready=False, restarts=5, waiting="CrashLoopBackOff", last_term="OOMKilled"
                )
            ],
        ),
        pod(
            "pull", phase="Pending", containers=[container(ready=False, waiting="ImagePullBackOff")]
        ),
    ]
    res = run("k8s_find_unhealthy_pods", {}, apis(pods))
    assert res.status == "ok"
    bad = {p["name"]: p["reasons"] for p in res.data["unhealthy_pods"]}
    assert set(bad) == {"crash", "pull"}
    assert any("CrashLoopBackOff" in r for r in bad["crash"])
    assert any("OOMKilled" in r for r in bad["crash"])
    assert any("ImagePullBackOff" in r for r in bad["pull"])


def test_k8s_healthy_returns_empty_list():
    res = run("k8s_find_unhealthy_pods", {}, apis([pod("a"), pod("b")]))
    assert res.data["unhealthy_pods"] == []


def test_k8s_unauthorized_namespace_never_calls_client():
    ca = apis()
    res = run("k8s_list_pods", {"namespace": "kube-system"}, ca)
    assert res.status == "rejected"
    ca[0].list_namespaced_pod.assert_not_called()
    assert run("k8s_find_unhealthy_pods", {"namespace": "default"}, ca).status == "rejected"


def test_k8s_restart_requires_confirmation():
    core, apps = apis()
    ca = (core, apps)
    res = run("k8s_restart_deployment", {"deployment": "prediction-api"}, ca)
    assert res.status == "pending_confirmation"
    apps.patch_namespaced_deployment.assert_not_called()
    with patch("agent.tools.kubernetes_tools.get_apis", lambda: ca):
        done = registry.confirm(res.data["action_id"])
    assert done.status == "ok"
    assert apps.patch_namespaced_deployment.call_count == 1
    body = apps.patch_namespaced_deployment.call_args.args[2]
    assert "kubectl.kubernetes.io/restartedAt" in str(body)


def test_k8s_scale_bounds_and_confirmation():
    ca = apis()
    assert run("k8s_scale_deployment", {"replicas": 99}, ca).status == "error"
    res = run("k8s_scale_deployment", {"replicas": 3}, ca)
    assert res.status == "pending_confirmation"
    ca[1].patch_namespaced_deployment_scale.assert_not_called()


def test_k8s_403_maps_to_rbac_error():
    ca = apis(pods_exc=ApiException(status=403, reason="Forbidden"))
    res = run("k8s_list_pods", {}, ca)
    assert res.status == "error" and "forbidden by RBAC" in res.error


def test_k8s_connection_error_maps_to_unavailable():
    ca = apis(pods_exc=MaxRetryError(None, "http://x", "Connection refused"))
    assert run("k8s_list_pods", {}, ca).status == "unavailable"


def test_k8s_get_deployed_version():
    core, apps = MagicMock(), MagicMock()
    apps.read_namespaced_deployment.return_value = NS(
        metadata=NS(labels={"app.kubernetes.io/version": "1.0.0"}),
        spec=NS(
            replicas=2,
            template=NS(
                spec=NS(
                    containers=[
                        NS(
                            image="mlops-inference:1.0.0",
                            env_from=[NS(config_map_ref=NS(name="inference-config"))],
                        )
                    ]
                )
            ),
        ),
        status=NS(ready_replicas=2),
    )
    core.read_namespaced_config_map.return_value = NS(
        data={"APP_VERSION": "1.0.0", "MODEL_VERSION": "3"}
    )
    res = run("k8s_get_deployed_version", {}, (core, apps))
    assert res.data["image_tags"] == ["1.0.0"] and res.data["configmap"]["MODEL_VERSION"] == "3"


def test_k8s_diagnose_service_flags_zero_endpoints():
    core, apps = MagicMock(), MagicMock()
    core.read_namespaced_service.return_value = NS(
        metadata=NS(name="prediction-api"),
        spec=NS(
            type="ClusterIP",
            cluster_ip="10.0.0.1",
            ports=[NS(port=80, target_port=8000)],
            selector={"app": "x"},
        ),
    )
    core.read_namespaced_endpoints.return_value = NS(subsets=None)
    core.list_namespaced_pod.return_value = NS(
        items=[
            pod(
                "p1",
                phase="Pending",
                containers=[container(ready=False, waiting="ImagePullBackOff")],
            )
        ]
    )
    core.read_namespaced_pod_log.side_effect = ApiException(status=400)
    core.list_namespaced_event.return_value = NS(
        items=[
            NS(
                type="Warning",
                reason="Failed",
                message="Failed to pull image",
                count=3,
                involved_object=NS(kind="Pod", name="p1"),
                last_timestamp=datetime.now(UTC),
                event_time=None,
                metadata=NS(creation_timestamp=datetime.now(UTC)),
            )
        ]
    )
    res = run("k8s_diagnose_service", {}, (core, apps))
    text = " ".join(res.data["findings"])
    assert "0 ready endpoints" in text and "ImagePullBackOff" in text and "Failed to pull" in text
