"""Importing this package registers every tool with the global registry."""

from agent.tools import (  # noqa: F401
    airflow_tools,
    dvc_tools,
    kubernetes_tools,
    mlflow_tools,
    policy_tools,
)
