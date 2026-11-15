"""Capture UI screenshots of the live stack with headless Edge (Playwright, no browser download).

Run:  uv run --with playwright python -m scripts.take_screenshots
Logs in to the local Airflow with the credentials in .env (never printed)."""

import html
import json
import pathlib

from playwright.sync_api import sync_playwright

from agent.config import get_settings

OUT = pathlib.Path("screenshots")
MLFLOW = "http://localhost:5000"
AIRFLOW = "http://localhost:8080"


def text_page(title: str, path: str) -> str:
    body = html.escape(pathlib.Path(path).read_text(encoding="utf-8"))
    return (
        "<body style='background:#0d1117;color:#c9d1d9;font:14px Consolas,monospace;padding:20px'>"
        f"<h3 style='color:#58a6ff'>{html.escape(title)}</h3><pre>{body}</pre></body>"
    )


def main() -> None:
    s = get_settings()
    from mlflow.tracking import MlflowClient

    c = MlflowClient(MLFLOW)
    exp = c.get_experiment_by_name("breast-cancer-classifier").experiment_id
    best = c.search_runs([exp], order_by=["metrics.f1 DESC"], max_results=1)[0].info.run_id
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1500, "height": 900})

        def shot(url, name, wait=4000, full=False):
            pg.goto(url, wait_until="domcontentloaded")
            pg.wait_for_timeout(wait)
            pg.screenshot(path=str(OUT / name), full_page=full)
            print("saved", name)

        shot(f"{MLFLOW}/#/experiments", "mlflow_experiments.png")
        shot(f"{MLFLOW}/#/experiments/{exp}", "mlflow_runs_table.png", 6000)
        shot(f"{MLFLOW}/#/experiments/{exp}/runs/{best}", "mlflow_run_tags_provenance.png", 6000, True)
        shot(f"{MLFLOW}/#/models/breast_cancer_clf", "mlflow_model_registry.png", 5000)

        pg.goto(f"{AIRFLOW}/", wait_until="domcontentloaded")
        pg.wait_for_timeout(1500)
        if pg.locator("input[name=username]").count():
            pg.fill("input[name=username]", s.airflow_username)
            pg.fill("input[name=password]", s.airflow_password)
            pg.click("button[type=submit]")
            pg.wait_for_timeout(4000)
        shot(f"{AIRFLOW}/dags/ml_training_pipeline", "airflow_dag_overview.png", 5000)
        pg.goto(f"{AIRFLOW}/dags/ml_training_pipeline", wait_until="domcontentloaded")
        pg.wait_for_timeout(4000)
        pg.mouse.click(145, 83)  # graph view toggle (left pane)
        pg.wait_for_timeout(4000)
        pg.screenshot(path=str(OUT / "airflow_dag_graph.png"))
        print("saved airflow_dag_graph.png")
        shot(f"{AIRFLOW}/dags/ml_training_pipeline/runs", "airflow_runs_list.png", 5000)
        shot(f"{AIRFLOW}/dags/ml_training_pipeline/tasks", "airflow_tasks.png", 5000)
        b.close()

        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pages = {
            "k8s_get_all.png": ("kubectl -n mlops-app get all", "evidence/k8s/get_all.txt"),
            "k8s_describe_deployment.png": ("kubectl describe deploy prediction-api",
                                            "evidence/k8s/describe_deployment.txt"),
            "k8s_rbac_can_i.png": ("kubectl auth can-i ... --as=mlops-agent",
                                   "evidence/k8s/rbac_can_i.txt"),
            "k8s_failure_A_imagepull.png": ("Failure A: pods and events",
                                            "evidence/failures/A/kubectl_get.txt"),
            "k8s_failure_B_readiness.png": ("Failure B: pods and endpoints",
                                            "evidence/failures/B/kubectl_get.txt"),
            "dvc_versions_and_checkout.png": ("DVC versions, diff, status, md5 checkout",
                                              "evidence/dvc/versions.md"),
            "dvc_dag.png": ("dvc dag", "evidence/dvc/dvc_dag.txt"),
        }
        for name, (title, src) in pages.items():
            pg.set_content(text_page(title + "  (real output saved in " + src + ")", src))
            pg.screenshot(path=str(OUT / name), full_page=True)
            print("saved", name)
        b.close()
    json.dump({"note": "k8s_*/dvc_* screenshots render real saved command output"},
              open(OUT / "README.json", "w"))


if __name__ == "__main__":
    main()
