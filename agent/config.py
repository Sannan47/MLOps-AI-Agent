"""Environment-driven configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    anthropic_api_key: str = ""
    agent_model: str = "claude-opus-5-5"
    llm_provider: str = "anthropic"  # anthropic | gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    mlflow_tracking_uri: str = "http://localhost:5000"
    airflow_base_url: str = "http://localhost:8080"
    airflow_username: str = ""
    airflow_password: str = ""
    k8s_namespace: str = "mlops-app"
    kubeconfig: str = ""
    approved_dags: str = "ml_training_pipeline"
    tool_timeout_seconds: float = 20.0
    dvc_repo_path: str = "."
    save_transcripts: bool = False
    action_ttl_seconds: int = Field(default=600)

    @property
    def allowed_namespaces(self) -> list[str]:
        return [self.k8s_namespace]

    @property
    def approved_dag_list(self) -> list[str]:
        return [d.strip() for d in self.approved_dags.split(",") if d.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
