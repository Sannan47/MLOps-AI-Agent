from __future__ import annotations

import json

import joblib
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from serving.app import create_app


def make_model_dir(tmp_path):
    X = [[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 1.1]]
    y = [0, 1, 0, 1]
    joblib.dump(LogisticRegression().fit(X, y), tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model_version": "3", "run_id": "abc", "data_version": "data-v1.1"}), "utf-8"
    )
    return tmp_path / "model.joblib"


def test_serving_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(make_model_dir(tmp_path)))
    monkeypatch.setenv("APP_VERSION", "1.0.0")
    c = TestClient(create_app())
    assert c.get("/health/live").status_code == 200
    assert c.get("/health/ready").status_code == 200
    r = c.post("/predict", json={"features": [1.0, 1.0]})
    assert r.status_code == 200 and r.json()["prediction"] in (0, 1)
    assert c.post("/predict", json={"features": [1.0]}).status_code == 422  # wrong length
    assert c.post("/predict", json={"features": "bad"}).status_code == 422
    v = c.get("/version").json()
    assert v["app_version"] == "1.0.0" and v["model_version"] == "3" and v["run_id"] == "abc"


def test_serving_ready_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    c = TestClient(create_app())
    assert c.get("/health/live").status_code == 200
    r = c.get("/health/ready")
    assert r.status_code == 503 and "FileNotFoundError" in r.json()["reason"]
    assert c.post("/predict", json={"features": [1.0]}).status_code == 503
