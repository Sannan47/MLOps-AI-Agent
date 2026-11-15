| Tool | Module | Risk | Confirmation | Inputs | Backing |
|---|---|---|---|---|---|
| `airflow_list_dags` | airflow | read | no | - | Airflow 3 REST API v2 (JWT) |
| `airflow_get_dag_runs` | airflow | read | no | dag_id, limit | Airflow 3 REST API v2 (JWT) |
| `airflow_get_dag_run_status` | airflow | read | no | dag_id, run_id | Airflow 3 REST API v2 (JWT) |
| `airflow_trigger_dag` | airflow | mutate | yes | dag_id, conf | Airflow 3 REST API v2 (JWT) |
| `airflow_wait_for_dag_run` | airflow | read | no | dag_id, run_id, timeout_s | Airflow 3 REST API v2 (JWT) |
| `airflow_get_failed_tasks` | airflow | read | no | dag_id, run_id | Airflow 3 REST API v2 (JWT) |
| `airflow_get_task_logs` | airflow | read | no | dag_id, run_id, task_id, try_number, tail_lines | Airflow 3 REST API v2 (JWT) |
| `dvc_list_tracked_datasets` | dvc | read | no | - | git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags |
| `dvc_get_current_version` | dvc | read | no | - | git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags |
| `dvc_list_versions` | dvc | read | no | - | git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags |
| `dvc_compare_versions` | dvc | read | no | rev_a, rev_b, path | git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags |
| `dvc_check_data_changed` | dvc | read | no | - | git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags |
| `dvc_get_dataset_for_run` | dvc | read | no | run_id | git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags |
| `dvc_get_lineage` | dvc | read | no | - | git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags |
| `dvc_reproduce_dataset_version` | dvc | read | no | rev, verify | git/dvc subprocess (allowlisted, shell=False) + dvc.api + MLflow tags |
| `k8s_list_deployments` | k8s | read | no | namespace | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_list_pods` | k8s | read | no | namespace, label_selector | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_get_pod_status` | k8s | read | no | namespace, pod | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_get_pod_logs` | k8s | read | no | namespace, pod, container, tail_lines, previous | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_list_services` | k8s | read | no | namespace | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_get_events` | k8s | read | no | namespace, involved_object, warning_only | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_find_unhealthy_pods` | k8s | read | no | namespace | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_get_deployed_version` | k8s | read | no | namespace, deployment | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_diagnose_service` | k8s | read | no | namespace, service | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_restart_deployment` | k8s | mutate | yes | namespace, deployment | kubernetes Python client (CoreV1/AppsV1) |
| `k8s_scale_deployment` | k8s | mutate | yes | namespace, deployment, replicas | kubernetes Python client (CoreV1/AppsV1) |
| `mlflow_list_experiments` | mlflow | read | no | - | MlflowClient (tracking + registry) |
| `mlflow_list_runs` | mlflow | read | no | experiment, max_results, order_by, filter | MlflowClient (tracking + registry) |
| `mlflow_get_run_details` | mlflow | read | no | run_id | MlflowClient (tracking + registry) |
| `mlflow_find_best_run` | mlflow | read | no | experiment, metric, mode, max_results | MlflowClient (tracking + registry) |
| `mlflow_compare_runs` | mlflow | read | no | run_ids, last_n, metrics, experiment | MlflowClient (tracking + registry) |
| `mlflow_compare_experiments` | mlflow | read | no | exp_a, exp_b, metric | MlflowClient (tracking + registry) |
| `mlflow_get_model_registry` | mlflow | read | no | model_name | MlflowClient (tracking + registry) |
| `policy_evaluate_retraining` | policy | read | no | model_age_days, champion_f1, latest_data_version, trained_data_version, last_gate_failed, best_f1 | agent/policy.yaml |

Tools per module: {"airflow": 7, "dvc": 8, "k8s": 11, "mlflow": 7, "policy": 1}
