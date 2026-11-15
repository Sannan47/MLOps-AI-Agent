# Technical Report: Production-Ready MLOps AI Agent

Repository: https://github.com/Sannan47/MLOps-AI-Agent

## 1. Complete system architecture

See [architecture.md](../architecture.md) for the component, sequence, confirmation-flow, provenance and trust-boundary diagrams. Summary:

- **User interface:** FastAPI agent API (`/chat`, `/actions/{id}/confirm|reject`, `/health`, `/tools`) and a CLI (`python -m agent.main`).
- **Agent:** a manual tool-use loop (`agent/orchestrator.py`, max 12 iterations, time budget) over an `LLMClient` protocol. Implementations: `AnthropicLLM`, `GeminiLLM` and `OpenRouterLLM` (both OpenAI-compatible), and a scripted `FakeLLM` used by tests and CI so no API key is needed there.
- **Tool registry** (`agent/registry.py`): pydantic-validated inputs (`extra=forbid`), per-tool timeouts, permission checks, structured logging with secret redaction, and results as `ToolResult` (`ok | error | unavailable | timeout | rejected | pending_confirmation`). It never raises into the LLM loop.
- **Four tool modules** (33 tools in total, 7 to 11 per module) plus one policy helper:
  DVC (8), MLflow (7), Airflow (7), Kubernetes (11), policy (1).
- **ML system under test:** scikit-learn binary classifier (Breast Cancer Wisconsin), DVC-versioned dataset (`data-v1.0`, `data-v1.1`, `data-v2.0`), MLflow 3.18 tracking + registry with `champion`/`challenger` aliases, Airflow 3 DAG `ml_training_pipeline`, FastAPI inference service in a `kind` cluster (namespace `mlops-app`, 2 replicas).
- **Grounding:** a validator checks that numbers, run ids, commits, versions, pod names and `[E-xxxxxx]` citations in the final answer appear in an `ok` tool result from the same request; it re-asks once and then strips unsupported lines. If all tools of a module fail, the answer must say the information could not be retrieved.
- **Retraining policy:** `agent/policy.yaml` (model age, min F1, F1 drop vs champion, new data version, last gate failed). The agent may recommend retraining only from `policy_evaluate_retraining`, a failed gate, or supplied degradation results, and never triggers it on its own (a trigger also needs human confirmation).

## 2. Tool definitions and permissions

The full generated table (name, module, risk, confirmation, inputs, backing API) is in [tool_table.md](tool_table.md). Mutating tools: `airflow_trigger_dag`, `k8s_restart_deployment`, `k8s_scale_deployment` (replicas bounded 1 to 5); each only creates a pending action that a human must confirm.

**Allowlists and restrictions**
- Kubernetes: namespace `mlops-app` only, checked before any client call; no delete, exec or apply tools exist.
- Airflow: only `APPROVED_DAGS=ml_training_pipeline` can be triggered; authenticated with a JWT from `POST /auth/token`.
- DVC/Git: allowlist of `dvc diff|status|data status|list|get` and `git rev-parse|log|tag|describe|show`; `shell=False`; revisions validated by regex and `git rev-parse --verify`; `dvc get` only into a temporary directory.

**Kubernetes RBAC** (`kubernetes/rbac.yaml`, `evidence/k8s/rbac_can_i.txt`): namespaced Role; `get/list/watch` on pods, pod logs, services, endpoints, events, configmaps and deployments; `patch` on deployments and `deployments/scale`. Verified:

| Check as the agent ServiceAccount | Result |
|---|---|
| delete pods in `mlops-app` | no |
| list pods in `default` | no |
| create deployments in `mlops-app` | no |
| list pods / read logs in `mlops-app` | yes |
| patch deployments in `mlops-app` | yes |

The agent's kubeconfig (24 h token) lives outside the repository; a 403 is returned for other namespaces (`kube-system` verified).

## 3. Screenshots and exported evidence

Exported evidence (all produced by real runs on the live stack):

| Area | Files |
|---|---|
| DVC | `evidence/dvc/versions.md` (rows, columns, md5 per version; `dvc diff`, `dvc status -c`; reproducible checkout with matching md5), `dvc_dag.txt`, `metrics.txt` |
| MLflow | 9 runs in 2 experiments, versions 1 to 5 in `breast_cancer_clf` (champion = 1); tags verified in conversation 1 and by tests |
| Airflow | `evidence/airflow/success_run.json`, `failure_C_corrupted_dataset.json`, `gate_failure_min_f1.json` |
| Kubernetes | `evidence/k8s/` (`get_all.txt`, `describe_deployment.txt` with probes and limits, `configmap.yaml`, `service.txt`, `rbac_can_i.txt`, `kubeconform.txt`, `predict_via_service.json`) |
| Failures | `evidence/failures/A`, `B`, `C` |
| Conversations | `evidence/conversations/` |

Screenshots (UI shots are captured from the live MLflow and Airflow UIs by `scripts/take_screenshots.py`; the `k8s_*` and `dvc_*` images render the real command output saved under `evidence/`, they are not terminal captures):

- `screenshots/airflow_dag_overview.png`
- `screenshots/airflow_runs_list.png`
- `screenshots/airflow_tasks.png`
- `screenshots/dvc_dag.png`
- `screenshots/dvc_versions_and_checkout.png`
- `screenshots/k8s_describe_deployment.png`
- `screenshots/k8s_failure_A_imagepull.png`
- `screenshots/k8s_failure_B_readiness.png`
- `screenshots/k8s_get_all.png`
- `screenshots/k8s_rbac_can_i.png`
- `screenshots/mlflow_experiments.png`
- `screenshots/mlflow_model_registry.png`
- `screenshots/mlflow_run_tags_provenance.png`
- `screenshots/mlflow_runs_table.png`

## 4. Five tested agent conversations

All were run with the real agent against the live stack. Transcripts include the request id, ordered tool calls with arguments, statuses and durations, and the final answer with `[E-id]` citations. They were produced with Gemini (`gemini-3.5-flash`; `gemini-3-flash-preview` where noted) because the free tier was used. Conversations 6 to 9 (failures B and C) were recorded with Claude Haiku 5.5 through OpenRouter.

| # | Prompt | Transcript | Tools called (all `ok`) |
|---|---|---|---|
| 1 | "Which dataset version was used for the best run in experiment `breast-cancer-classifier`, and how can I reproduce it?" | `conv1_dvc_dataset_version.json` | `mlflow_find_best_run`, `dvc_get_dataset_for_run`, `dvc_reproduce_dataset_version` |
| 2 | "Compare the last five training runs and recommend the best candidate based on F1 score." | `conv2_mlflow_compare_last_five.json` | `mlflow_list_experiments`, `mlflow_list_runs`, `mlflow_compare_runs`, `mlflow_get_model_registry`, `dvc_get_current_version`, `mlflow_get_run_details`, `policy_evaluate_retraining` |
| 3 | "Trigger the model training pipeline and report whether execution succeeded." | `conv3a_airflow_trigger_request.json` (request, `pending_confirmation`), `conv3b_airflow_result_report.json` (after the user's confirmation) | `airflow_list_dags`, `airflow_trigger_dag` then `airflow_get_dag_runs`, `airflow_get_dag_run_status`, `airflow_get_failed_tasks`, `airflow_get_task_logs`, `policy_evaluate_retraining` |
| 4 | "Why is my prediction API unavailable?" (Failure A injected) | `conv4_k8s_why_unavailable_failureA.json` | `k8s_list_services`, `k8s_find_unhealthy_pods`, `k8s_list_deployments`, `k8s_get_events`, `k8s_get_deployed_version`, `k8s_list_pods`, `k8s_diagnose_service`, `mlflow_get_model_registry` |
| 5 | "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary." | `conv5_integration_scenario_newdata.json` | `mlflow_find_best_run`, `mlflow_get_model_registry`, `dvc_list_versions`, `dvc_get_dataset_for_run`, `k8s_list_deployments`, `k8s_find_unhealthy_pods`, `k8s_get_deployed_version`, `airflow_get_dag_runs`, `airflow_get_failed_tasks`, `mlflow_get_run_details`, `policy_evaluate_retraining` |

Spot-checked claims: conversation 1's dataset version (`data-v1.0`), commit (`88de5a5...`), md5 (`96631a6c...`) and the md5-verified `dvc get` match `evidence/dvc/versions.md`; conversation 3b's gate failure message (F1 0.8980 against champion 0.9244) matches the Airflow task log and MLflow.

**Notes on behaviour.** Conversation 3a shows the confirmation flow: the agent created a pending action and said it could not report success yet; a human then confirmed at the CLI, the DAG ran (failed at the quality gate on default data), and 3b reported the per-task result with the reason.

**Conversation 5.** Recorded with `gemini-3-flash-preview` (variant b: newer data available, so retraining is recommended). The agent called all four modules and the policy tool, produced the three required sections with citations, identified the best run (`d732b237...`, F1 0.924), its recorded dataset `data-v1.0` (md5 `96631a6c...`), the deployment (2/2 ready, serving champion v1) and the failed last pipeline run, and recommended retraining only because `policy_evaluate_retraining` reported a newer data version (`data-v2.0`). `airflow_trigger_dag` was not called. Variant (a) (policy not triggered) was not recorded: the champion was trained on `data-v1.0` while `data-v2.0` exists, and training on `data-v2.0` fails the quality gate (F1 0.898), so no champion on the latest data exists to demonstrate it live. The automated test `test_multi_tool_integration_scenario_calls_all_modules_and_no_trigger` covers the no-recommendation path.

Earlier attempts at this conversation were discarded and are not evidence: two hit the free-tier quota, one on a lite-tier model gave a fluent but wrong answer, and one exposed a bug in the grounding validator (multi-id citations were mistaken for numbers), which was fixed with a regression test and a stricter system-prompt recipe before the recorded run.

## 5. Failure investigations

Three failures (Kubernetes image tag, Kubernetes config, Airflow corrupted data), each with symptom, tool trace, root cause, resolution and prevention: [failure_investigations.md](failure_investigations.md).

## 6. Testing summary

- `uv run pytest`: **73 tests pass in about 20 s** with no network access; plus 3 slow tests (real MLflow/sklearn training: provenance tags, gate blocks promotion, champion/challenger promotion) and integration tests that need the live stack (DVC compare, MLflow, live cluster).
- Coverage of `agent/`: **85%** (gate 70% in CI).
- Required cases covered: successful tool selection and execution (FakeLLM tests), the mandatory multi-tool scenario with no auto-trigger, tool timeout, invalid arguments, service failures (MLflow unavailable, Airflow 500 and down), **no invented results when a tool is unavailable**, **unauthorized Kubernetes namespace rejected before any client call**, mutating actions executed only after confirmation (exactly once), expired/unknown action ids, non-approved DAG, structured log fields, secret redaction, DVC allowlist and bad revisions, best-run max/min/ties, readiness checks, failed-task and log diagnostics, unhealthy-pod detection (CrashLoopBackOff, ImagePullBackOff), retraining policy rules, grounding validator, validation and gate behaviour, serving endpoints (503 without model, 422 on bad input), Gemini adapter (schema reduction, retries, errors).
- CI: `.github/workflows/ci.yml` runs lint, tests with coverage, kubeconform and a Docker build on pull requests and pushes to `main`; pull request: https://github.com/Sannan47/MLOps-AI-Agent/pull/1 (CI result recorded in section 3 evidence once green).

Bugs found only by running against the live stack (fixed, with regression tests where possible): MLflow 3.18 Host-header check blocked Airflow; MLflow auto-detected `uv.lock` and exported conflicting pins; git rejected the mounted repo as not owned by the container user; a task parameter named `run_id` collided with an Airflow reserved key; the init container created an unwritable passwords file; the DAG recorded workspace data as provenance instead of the revision used (two mislabelled runs corrected); the log tool dropped structured exception details; the registry tool returned an empty list for an unknown model name, which misled a weaker LLM into claiming nothing was registered.

## 7. Known limitations and proposed improvements

Limitations
- Single-node `kind` cluster; SQLite MLflow backend; local-directory DVC remote; LocalExecutor Airflow; in-memory pending actions (lost on API restart); regex-based grounding validator (checks tokens, not reasoning; it cannot detect a wrong conclusion built from correct facts); no drift detection; LLM latency and cost; Gemini free tier (about 20 requests per day per model) limited how many conversations could be recorded and forced switching models.
- Conversation quality depends on the model: a lite-tier model produced a fluent, fully cited but wrong answer in a trial run (treated the latest run as the best, claimed nothing was registered). That transcript was discarded and the tool and prompt were fixed, but it shows the validator is not a substitute for a strong model.
- The unhealthy-pod check counts any restart, so pods look unhealthy after a node restart even when Ready.
- `AnthropicLLM` was implemented but not exercised against the live API.
- Failure A was recorded with Gemini, B and C with Claude Haiku 5.5 via OpenRouter (conversations 4, 6 to 9). A validator bug found while recording B (a pending action's id was treated as unsupported) was fixed and the conversation re-recorded.

Improvements
- S3/MinIO DVC remote, Postgres for MLflow, Celery/Kubernetes executor, persistent action store, OpenTelemetry tracing, drift monitoring (Evidently), HPA and PodDisruptionBudget, GitOps for image tags, model-in-image vs model registry loading, an LLM-as-judge or rule check for conclusions, restart-count decay in health checks.
