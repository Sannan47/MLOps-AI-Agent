# Failure investigations

Three failures were injected on the live stack, one per layer. Every number below comes from the saved evidence files, not from memory. "Agent conversation" means the real agent loop with a live LLM; "tool trace" means the same agent tools run directly (no LLM) when no conversation was recorded.

## Failure A: Kubernetes, broken image tag (infrastructure layer)

**Injection:** `kubectl -n mlops-app set image deploy/prediction-api api=mlops-inference:9.9.9-missing` (done by hand, not through the agent).

**Symptom:** rollout stuck; 2 new pods `ImagePullBackOff` / `ErrImagePull`, 1 old pod still Ready, so the Service had 1 ready endpoint instead of 2.

**Agent conversation:** `evidence/conversations/conv4_k8s_why_unavailable_failureA.json`, question "Why is my prediction API unavailable?" (request `ac07be2a3427`).

| Tool (evidence id) | What it showed |
|---|---|
| `k8s_find_unhealthy_pods` (E-21b8d9) | two pods Pending with `ImagePullBackOff` / `ErrImagePull` |
| `k8s_get_events` (E-50b95c) | `Failed to pull image "mlops-inference:9.9.9-missing": repository does not exist` |
| `k8s_get_deployed_version` (E-8cd83b) | image tag `9.9.9-missing` while the ConfigMap still says `APP_VERSION=1.0.0` |
| `k8s_list_services` / `k8s_diagnose_service` (E-35e83d, E-39324d) | 1 ready, 2 not-ready endpoints |
| `k8s_list_deployments` (E-48ad43) | 2 desired, 1 ready/available |

**Root cause:** the deployment references an image tag that does not exist in the cluster; the rolling update cannot complete and capacity is halved.

**Resolution:** `kubectl rollout undo deploy/prediction-api` (done by hand); rollout back to 2/2 Ready.

**Honest notes:** the agent's recommendation to "update the image and then request a restart" is correct about the tag but the restart is redundant after a rollout; it did not claim a restart alone would fix it. Raw outputs: `evidence/failures/A/`.

**Prevention:** pin image tags through CI/GitOps, use `kubectl rollout status` gates, and keep `maxUnavailable: 1` so a bad tag never takes the whole service down (this is why one pod kept serving).

## Failure B: Kubernetes, readiness failing from bad config (application layer)

**Injection:** `kubectl patch configmap inference-config` setting `MODEL_PATH=/models/missing.joblib`, then `rollout restart` (`scripts/inject_failure_bad_config.ps1`).

**Symptom:** new pods Running but not Ready; Service with 1 ready and 2 not-ready endpoints.

**Tool trace** (`evidence/failures/B/tool_results.json`, `kubectl_get.txt`; no LLM conversation recorded for this failure):
- events: `Readiness probe failed: HTTP probe failed with statuscode: 503`
- pod logs (`k8s_diagnose_service`): `model load failed: FileNotFoundError: [Errno 2] No such file or directory: '/models/missing.joblib'`
- `k8s_get_deployed_version`: image tag `1.0.0` (correct) but ConfigMap `MODEL_PATH=/models/missing.joblib`

**Root cause:** the application starts but cannot load the model file named in the ConfigMap, so `/health/ready` returns 503. The image is fine; this is a configuration error, so only the config (not the image) must change.

**Resolution through the agent's confirmation flow:** ConfigMap fixed by `kubectl apply -f kubernetes/configmap.yaml`; then the agent requested `k8s_restart_deployment` (status `pending_confirmation`, nothing executed), the user confirmed, and exactly one patch was sent (`evidence/failures/B/restart_confirmation.json`). Rollout completed with 2/2 Ready.

**Prevention:** validate config at deploy time (a startup check or admission policy for `MODEL_PATH`), and keep readiness separate from liveness so the failing pods are never sent traffic (they were not).

## Failure C: Airflow, corrupted dataset (data/workflow layer)

**Injection:** `scripts/make_dataset_versions.py corrupted` (label column removed, 35% nulls in a feature, never tagged), then trigger `ml_training_pipeline` with `{"dataset_path": "data/raw/corrupted.csv"}` through the confirmation flow.

**Symptom:** DAG run `failed`.

**Tool trace** (`evidence/airflow/failure_C_corrupted_dataset.json`):
- `airflow_get_failed_tasks`: `validate_dataset` = failed; `preprocess_data`, `train_model`, `evaluate_and_log_mlflow`, `apply_quality_gate` = `upstream_failed`
- `airflow_get_task_logs`: `DataValidationError: column(s) ['label'] missing`
- MLflow registry before and after: identical versions `['1','2','3','4']`, champion = 1 (promotion blocked)

**Root cause:** the dataset failed validation; the DAG raises `AirflowFailException` for this deterministic error so it fails once without wasteful retries, and the default `all_success` trigger rule keeps every downstream task from running.

**Resolution:** use a valid DVC revision (`data-v1.0`, `dvc checkout`/`dvc get --rev`) and re-trigger with confirmation. Verified: the run with `{"dataset_rev": "data-v1.0"}` succeeded, all 5 tasks green, run recorded `data.version=data-v1.0` (`evidence/airflow/success_run.json`).

**Prevention:** the validation gate itself; plus schema tests on new data versions before tagging.

## Related: quality-gate failure (not one of A/B/C)

Trigger with `{"dataset_rev": "data-v1.1", "min_f1": 0.999}`: all tasks succeed until `apply_quality_gate` fails (`f1 0.8980 < min_f1 0.999 ...`); registry and champion unchanged (`evidence/airflow/gate_failure_min_f1.json`). The agent reported a real gate failure of the same kind (F1 0.8980 against champion 0.9244) in `conv3b_airflow_result_report.json`.

## Replay tests

Fixture-based replay tests of these traces (`tests/`) are covered by the unit tests for `find_unhealthy_pods` (CrashLoopBackOff, ImagePullBackOff), `diagnose_service` (0 endpoints), log diagnostics extraction and the validation/gate paths; a dedicated replay file built from the saved JSON is not included.
