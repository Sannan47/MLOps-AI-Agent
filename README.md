# MLOps AI Agent

A conversational agent that answers operational questions about an ML system and runs authorized operations, using **DVC, MLflow, Apache Airflow 3 and Kubernetes** through structured tools. It cites evidence for every operational claim, never changes anything without explicit user confirmation, and diagnoses failures across the data, workflow and infrastructure layers.

- Architecture, diagrams and trust boundaries: [architecture.md](architecture.md)
- Technical report: [docs/technical_report.md](docs/technical_report.md)
- Failure investigations: [docs/failure_investigations.md](docs/failure_investigations.md)
- Tool definitions and permissions: [docs/tool_table.md](docs/tool_table.md)
- Environment versions: [docs/environment.md](docs/environment.md)

## Repository layout

```
agent/        orchestrator, LLM clients, tool registry, permissions, policy, tools/{dvc,mlflow,airflow,kubernetes}_tools.py
api/          FastAPI agent interface (/chat, /actions, /health, /tools)
serving/      FastAPI inference service deployed to Kubernetes
training/     validate, preprocess, train, evaluate, quality_gate (shared by DVC and Airflow)
dags/         Airflow 3 DAG `ml_training_pipeline`
kubernetes/   namespace, configmap, deployment, service, RBAC
infra/        docker-compose (Airflow + MLflow), kind config, Airflow Dockerfile
scripts/      dataset versions, history population, model export, failure injection, evidence collection
tests/        pytest suite (all external services mocked by default)
evidence/     real tool outputs, failure traces and saved agent conversations
```

## Prerequisites

Docker Desktop (WSL2 backend, at least 6 GB RAM), `kubectl`, `kind`, `uv`, Python 3.12, Git. Exact versions used are in [docs/environment.md](docs/environment.md). Docker's disk image grows to ~20 GB; put it on a drive with space (Docker Desktop > Settings > Resources > Disk image location).

## Installation

```bash
uv sync
```

## Environment configuration

```bash
cp .env.example .env     # then edit .env; it is git-ignored
```

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `anthropic` (default) or `gemini` |
| `ANTHROPIC_API_KEY` / `AGENT_MODEL` | Claude key and model (default `claude-opus-5-5`) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Gemini key and model (free tier has small daily quotas per model) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Airflow API credentials (choose your own password) |
| `K8S_NAMESPACE`, `KUBECONFIG` | approved namespace and the agent's restricted kubeconfig |
| `APPROVED_DAGS` | DAGs the agent may trigger (`ml_training_pipeline`) |
| `TOOL_TIMEOUT_SECONDS`, `DVC_REPO_PATH`, `SAVE_TRANSCRIPTS` | tool timeout, repo path, save conversations to `evidence/conversations/` |

Never commit `.env` or kubeconfigs.

## Launching each service (in order)

```bash
# 1. Kubernetes cluster
kind create cluster --config infra/kind-config.yaml

# 2. MLflow + Airflow (Docker Compose). Needs AIRFLOW_USERNAME/PASSWORD in .env
docker compose -f infra/docker-compose.yaml --env-file .env up -d --build
#    MLflow UI http://localhost:5000, Airflow UI http://localhost:8080

# 3. Data (DVC local remote, versions tagged data-v1.0, data-v1.1, data-v2.0)
uv run dvc pull

# 4. Initial training history (9 runs across dataset versions, registers champion/challenger)
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m scripts.populate_history

# 5. Build the inference image from the champion model and load it into kind
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m scripts.export_model
docker build -t mlops-inference:1.0.0 --build-arg APP_VERSION=1.0.0 .
kind load docker-image mlops-inference:1.0.0 --name mlops

# 6. Deploy
kubectl apply -f kubernetes/
kubectl -n mlops-app rollout status deploy/prediction-api

# 7. Agent credentials for the cluster (short-lived, namespaced, outside the repo)
kubectl -n mlops-app create token mlops-agent --duration=24h
#    put it in ~/.kube/mlops-agent.kubeconfig (see docs/environment.md) and point KUBECONFIG at it

# 8. Agent API
uv run uvicorn api.main:app --port 8001
```

The Airflow DAG starts paused: unpause it once in the UI (or `PATCH /api/v2/dags/ml_training_pipeline {"is_paused": false}`).

## Running tests

```bash
uv run pytest                          # unit tests, no network, ~20 s
uv run pytest -m slow                  # real MLflow/sklearn training tests (~90 s)
RUN_INTEGRATION=1 uv run pytest -m integration   # needs the live stack
uv run pytest --cov=agent --cov-fail-under=70
```

## Interacting with the agent

CLI (prompts `Confirm action <id>? [y/N]` for mutating actions):

```bash
uv run python -m agent.main "Compare the last five training runs and recommend the best candidate based on F1 score."
```

HTTP:

```bash
curl -s localhost:8001/chat -H "Content-Type: application/json" \
  -d '{"message":"Trigger the model training pipeline and report whether execution succeeded."}'
curl -s localhost:8001/actions                       # pending actions
curl -s -X POST localhost:8001/actions/<id>/confirm  # only a human can confirm
curl -s -X POST localhost:8001/actions/<id>/reject
curl -s localhost:8001/health                        # per-service reachability
curl -s localhost:8001/tools                         # tools with risk level
```

The LLM has no confirm tool. A mutating tool call only creates a pending action; the user approves it.

## Failure injection

```powershell
scripts/inject_failure_bad_image.ps1        # Failure A: image tag missing (ImagePullBackOff)
scripts/inject_failure_bad_config.ps1       # Failure B: bad MODEL_PATH (readiness 503)
bash scripts/inject_failure_corrupted_dataset.sh   # Failure C: corrupted dataset for the Airflow DAG
```

Then ask: "Why is my prediction API unavailable?" See [docs/failure_investigations.md](docs/failure_investigations.md).

## Troubleshooting

- **Docker "disk full" / engine will not start**: free space on C: or move the disk image (see Prerequisites).
- **Docker says WSL is too old**: from an administrator shell run `wsl --install --no-distribution --web-download`, then reboot.
- **Airflow API server crashes with a permission error**: the passwords file must be writable by user 50000 (handled in `infra/docker-compose.yaml`).
- **MLflow 403 "Invalid Host header" from Airflow**: the server needs `--allowed-hosts mlflow:5000,...` (set in compose).
- **Gemini 429 / quota**: free tiers allow ~20 requests/day per model. Switch `GEMINI_MODEL` or the key's project.
- **Kubernetes tools return `ConfigException`**: `KUBECONFIG` in `.env` must be a forward-slash absolute path.
