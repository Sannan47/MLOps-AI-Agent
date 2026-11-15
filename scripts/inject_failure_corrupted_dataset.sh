#!/usr/bin/env bash
# Failure C: create a corrupted dataset variant (never tagged) for the Airflow validation failure.
set -euo pipefail
uv run python scripts/make_dataset_versions.py corrupted --out data/raw/corrupted.csv
echo "Trigger the DAG with conf: {\"dataset_path\": \"data/raw/corrupted.csv\"}"
