# Inference service image. Build after `python scripts/export_model.py` populates serving/model/.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serving.txt

FROM python:3.12-slim
ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION} \
    MODEL_PATH=/app/serving/model/model.joblib \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY serving/__init__.py serving/app.py serving/
COPY serving/model serving/model
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live').status==200 else 1)"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
