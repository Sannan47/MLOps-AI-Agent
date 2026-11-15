# Architecture

## Components (Figure 1)

```mermaid
flowchart LR
  U[User] -->|question| API[FastAPI agent API<br/>/chat /actions /health]
  CLI[CLI agent.main] --> AG
  API --> AG
  subgraph AG[Agent]
    LLM[LLM client<br/>Gemini / OpenRouter / Claude] <--> ORC[Orchestrator loop<br/>tool selection, tracing]
    ORC --> REG[Tool registry<br/>schema validation, timeouts,<br/>permissions, structured logs]
    ORC --> GV[Grounding validator<br/>unavailable-tool guard]
  end
  REG --> DVC[DVC tools<br/>git/dvc allowlist, lineage]
  REG --> MLF[MLflow tools<br/>MlflowClient]
  REG --> AFL[Airflow tools<br/>REST API v2 + JWT]
  REG --> K8S[Kubernetes tools<br/>namespaced RBAC]
  REG --> POL[Policy tool<br/>policy.yaml]
  DVC --> GIT[(Git tags data-vX + DVC remote)]
  MLF --> MLS[(MLflow server<br/>runs + registry)]
  AFL --> AFS[(Airflow<br/>ml_training_pipeline)]
  K8S --> CL[(kind cluster<br/>mlops-app)]
  GV -->|answer with E-id citations| U
```

The agent never answers operational questions from model knowledge. Every fact must come from a tool result of the current request and is cited as `[E-xxxxxx]`.

## Mandatory scenario (sequence)

```mermaid
sequenceDiagram
  participant U as User
  participant A as Agent (LLM loop)
  participant M as MLflow
  participant D as DVC
  participant K as Kubernetes
  participant F as Airflow
  participant P as Policy
  U->>A: best recent model, its dataset, deployment status, retraining needed?
  A->>M: mlflow_find_best_run, mlflow_get_model_registry
  A->>D: dvc_get_dataset_for_run(best run), dvc_list_versions
  A->>K: k8s_get_deployed_version, k8s_find_unhealthy_pods
  A->>F: airflow_get_dag_runs(ml_training_pipeline)
  A->>P: policy_evaluate_retraining(facts)
  A->>A: grounding validator (numbers, ids, citations)
  A-->>U: Known facts / Uncertainties / Recommended actions (no auto-retrain)
```

## Confirmation flow for mutating actions

```mermaid
sequenceDiagram
  participant L as LLM
  participant R as Registry
  participant S as Pending store (TTL 10 min)
  participant H as Human (API/CLI)
  participant T as Airflow / K8s
  L->>R: airflow_trigger_dag / k8s_restart / k8s_scale
  R->>R: validate args, check namespace + DAG allowlist
  R->>S: create PendingAction (not executed)
  R-->>L: status=pending_confirmation
  H->>R: POST /actions/{id}/confirm
  R->>S: claim once (expired/unknown/used rejected)
  R->>R: re-check allowlists
  R->>T: execute stored args exactly as stored
  H-->>R: or POST /actions/{id}/reject
```

The LLM has no tool that confirms. Delete, exec and apply tools do not exist.

## Provenance (data and model lineage)

```mermaid
flowchart LR
  T[Git tag data-v1.0 / v1.1 / v2.0] --> DV[data/raw/dataset.csv.dvc md5]
  DV --> TR[training run<br/>capture_provenance]
  TR --> TG[MLflow run tags<br/>data.version, data.dvc_md5,<br/>data.git_commit, data.dvc_remote,<br/>code.git_commit, pipeline.trigger,<br/>airflow.dag_run_id]
  TG --> REG2[Model registry<br/>champion / challenger alias]
  REG2 --> IMG[export_model.py -> Docker image<br/>metadata.json]
  IMG --> KD[Deployment label + ConfigMap<br/>APP_VERSION / MODEL_VERSION]
  TG -. dvc_get_dataset_for_run .-> A2[Agent answers from recorded tags]
```

`dvc_get_dataset_for_run` reads the tags recorded at training time, not the current workspace.

## Trust boundaries

| Boundary | Control |
|---|---|
| LLM to tools | Pydantic `extra=forbid` schemas, per-tool timeouts, errors returned as `ToolResult` (never raised), tool list is closed |
| Mutations | Two-phase confirmation; only the human API/CLI can confirm; allowlists re-checked at execution |
| Kubernetes | Namespace allowlist (`mlops-app`) checked before any API call, plus a namespaced Role: no delete/exec, no other namespaces, `patch` only on deployments/scale |
| Airflow | JWT from `/auth/token`; only `APPROVED_DAGS` can be triggered |
| DVC / Git | Command allowlist, `shell=False`, revision regex + `git rev-parse --verify`, `dvc get` only into a scratch directory |
| Secrets | `.env` only; logs redact keys, tokens, bearer values, `sk-ant-` strings and JWTs; kubeconfig lives outside the repo |
| Answers | Grounding validator: numbers, ids, versions, pod names and citations must appear in an `ok` tool result; otherwise re-ask once, then strip |

## Pipeline (Figure 2)

`validate_dataset >> preprocess_data >> train_model >> evaluate_and_log_mlflow >> apply_quality_gate` with retries (2, exponential backoff). Deterministic failures (validation, gate) raise `AirflowFailException` so they do not retry; downstream tasks become `upstream_failed` and no model is registered or aliased.
