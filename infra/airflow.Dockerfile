FROM apache/airflow:3.0.6
USER airflow
RUN pip install --no-cache-dir scikit-learn pandas mlflow dvc matplotlib joblib pyyaml
RUN git config --global --add safe.directory '*'
