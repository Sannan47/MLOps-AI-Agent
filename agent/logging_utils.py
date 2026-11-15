"""Structured JSON logging, request-id propagation and secret redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_KEY_RE = re.compile(r"(?i)(key|token|secret|password|authorization|bearer)")
_VALUE_RES = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]+"),
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-~+/]+=*"),
]
MASK = "***"


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def redact_text(text: str) -> str:
    for rx in _VALUE_RES:
        text = rx.sub(MASK, text)
    return text


def redact(obj: Any) -> Any:
    """Recursively mask sensitive keys and secret-looking values."""
    if isinstance(obj, dict):
        return {k: (MASK if _KEY_RE.search(str(k)) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "log"),
            "request_id": getattr(record, "request_id", request_id_var.get()),
            "message": redact_text(record.getMessage()),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(redact(extra))
        return json.dumps(payload, default=str)


def get_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    get_logger().log(
        level,
        event,
        extra={"event": event, "fields": fields, "request_id": request_id_var.get()},
    )
