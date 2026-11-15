"""Generate docs/tool_table.md from the live tool registry."""

import json

import agent.tools  # noqa: F401
from agent.registry import registry

BACKING = {
    "dvc": "git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags",
    "mlflow": "MlflowClient (tracking + registry)",
    "airflow": "Airflow 3 REST API v2 (JWT)",
    "k8s": "kubernetes Python client (CoreV1/AppsV1)",
    "policy": "agent/policy.yaml",
}
rows = ["| Tool | Module | Risk | Confirmation | Inputs | Backing |", "|---|---|---|---|---|---|"]
for t in registry.tools.values():
    props = t.input_model.model_json_schema().get("properties", {})
    inputs = ", ".join(props) or "-"
    conf = "yes" if t.risk == "mutate" else "no"
    rows.append(
        f"| `{t.name}` | {t.module} | {t.risk} | {conf} | {inputs} | {BACKING.get(t.module, '')} |"
    )
counts = {k: len(v) for k, v in registry.by_module().items()}
out = "\n".join(rows) + f"\n\nTools per module: {json.dumps(counts)}\n"
open("docs/tool_table.md", "w", encoding="utf-8").write(out)
print(out[-200:])
