"""Run the agent's read-only K8s tools against the live cluster and save JSON evidence."""

import json
import sys
import time

import agent.tools  # noqa: F401
from agent.registry import registry


def snapshot(out_dir: str) -> dict:
    res = {}
    for name in (
        "k8s_find_unhealthy_pods",
        "k8s_get_deployed_version",
        "k8s_list_deployments",
        "k8s_list_services",
        "k8s_get_events",
        "k8s_diagnose_service",
    ):
        r = registry.execute(name, {}, "evidence")
        res[name] = r.model_dump(mode="json")
    with open(f"{out_dir}/tool_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    return res


if __name__ == "__main__":
    time.sleep(float(sys.argv[2]) if len(sys.argv) > 2 else 0)
    r = snapshot(sys.argv[1])
    for k, v in r.items():
        print(k, v["status"])
    print(
        "unhealthy:",
        [p["name"] for p in (r["k8s_find_unhealthy_pods"]["data"] or {}).get("unhealthy_pods", [])],
    )
    print("tags:", (r["k8s_get_deployed_version"]["data"] or {}).get("image_tags"))
    print("findings:", (r["k8s_diagnose_service"]["data"] or {}).get("findings", [])[:6])
