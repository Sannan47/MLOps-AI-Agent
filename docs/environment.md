# Environment

Host: Windows 10 Pro (10.0.19045). Everything below was run on this machine.

| Component | Version |
|---|---|
| Python | 3.12 (managed by `uv` 0.13.0) |
| Docker Desktop / Engine | 4.99.0 / 29.9.3 (WSL2 backend, 6 GB RAM) |
| kubectl | v1.37.2 |
| kind | v0.33.0 |
| Git | 2.56.0 (Windows) |
| DVC | 3.67.1 |
| MLflow (client and server) | 3.18.0 |
| Apache Airflow | 3.0.6 (LocalExecutor, simple auth manager, REST API v2) |
| PostgreSQL (Airflow) | 16 |
| scikit-learn / pandas / FastAPI / pydantic | resolved in `uv.lock` |
| LLM (agent) | Gemini via the OpenAI-compatible endpoint (`gemini-3.x-flash`) for the recorded conversations; Anthropic Claude supported through `LLM_PROVIDER=anthropic` (not exercised against the live API) |

Docker's disk image was moved to the E: drive (C: ran out of space).

## Agent kubeconfig (no secrets in Git)

The agent uses a namespaced ServiceAccount token, not admin credentials:

```bash
SERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')
kubectl config view --minify --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}' | base64 -d > ca.crt
TOKEN=$(kubectl -n mlops-app create token mlops-agent --duration=24h)
kubectl --kubeconfig ~/.kube/mlops-agent.kubeconfig config set-cluster kind-mlops --server=$SERVER --certificate-authority=ca.crt --embed-certs=true
kubectl --kubeconfig ~/.kube/mlops-agent.kubeconfig config set-credentials mlops-agent --token="$TOKEN"
kubectl --kubeconfig ~/.kube/mlops-agent.kubeconfig config set-context mlops-agent --cluster=kind-mlops --user=mlops-agent --namespace=mlops-app
kubectl --kubeconfig ~/.kube/mlops-agent.kubeconfig config use-context mlops-agent
```

Set `KUBECONFIG=C:/Users/<you>/.kube/mlops-agent.kubeconfig` in `.env` (forward slashes). The token expires after 24 h; regenerate it.

## Measured facts

- Inference image `mlops-inference:1.0.0`: 561 MB (scikit-learn + numpy + scipy), runs as non-root UID 10001, `HEALTHCHECK` healthy.
- Cluster: 2/2 `prediction-api` pods Ready, kubeconform: 7 resources valid (`evidence/k8s/kubeconform.txt`).
