"""Tool registry: metadata, JSON schemas, validated + timed + permission-checked execution."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from agent.config import get_settings
from agent.logging_utils import log_event, redact, request_id_var
from agent.permissions import NamespacePolicy, PendingStore
from agent.schemas import ToolResult

MAX_TIMEOUT_S = 900


class ToolRejected(Exception):
    """Raised by a tool to signal a policy rejection."""


class ServiceUnavailable(Exception):
    """Raised by a tool when its backing service cannot be reached."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any], Any]
    risk: Literal["read", "mutate"]
    module: str
    timeout_s: float | None = None


_STRIP = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength"}


def _clean_schema(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k not in _STRIP}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def _is_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, ServiceUnavailable | ConnectionError | httpx.TransportError):
        return True
    name = type(exc).__name__
    if name in {"MaxRetryError", "NewConnectionError", "ProtocolError", "ConnectTimeoutError"}:
        return True
    msg = str(exc)
    if any(
        s in msg
        for s in (
            "Max retries exceeded",
            "Connection refused",
            "ConnectionError",
            "Failed to establish a new connection",
        )
    ):
        return True
    cause = exc.__cause__ or exc.__context__
    return cause is not None and cause is not exc and _is_unavailable(cause)


class Registry:
    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}
        self.policy = NamespacePolicy()
        self.store = PendingStore()

    # ---- registration ----
    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def by_module(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for t in self.tools.values():
            out.setdefault(t.module, []).append(t.name)
        return out

    def anthropic_tool_specs(self) -> list[dict]:
        specs = []
        for t in self.tools.values():
            schema = _clean_schema(t.input_model.model_json_schema())
            schema["additionalProperties"] = False
            specs.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": schema,
                    "strict": True,
                }
            )
        return specs

    # ---- execution ----
    def _result(
        self, tool: str, status: str, request_id: str, started: float, **kw: Any
    ) -> ToolResult:
        res = ToolResult(
            tool=tool,
            status=status,  # type: ignore[arg-type]
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            evidence_id="E-" + uuid.uuid4().hex[:6],
            request_id=request_id,
            **kw,
        )
        return res

    def _log(self, res: ToolResult, spec: ToolSpec | None, args: Any) -> None:
        log_event(
            "tool_call",
            tool=res.tool,
            module=spec.module if spec else None,
            status=res.status,
            duration_ms=res.duration_ms,
            evidence_id=res.evidence_id,
            args_summary=redact(args) if isinstance(args, dict) else str(args)[:200],
        )

    def _run(self, spec: ToolSpec, inp: BaseModel, request_id: str, started: float) -> ToolResult:
        timeout = min(spec.timeout_s or get_settings().tool_timeout_seconds, MAX_TIMEOUT_S)
        ex = ThreadPoolExecutor(max_workers=1)
        token = request_id_var.set(request_id)
        try:
            fut = ex.submit(spec.fn, inp)
            data = fut.result(timeout=timeout)
            return self._result(spec.name, "ok", request_id, started, data=data)
        except FutureTimeout:
            return self._result(
                spec.name, "timeout", request_id, started, error=f"timed out after {timeout}s"
            )
        except ToolRejected as e:
            return self._result(spec.name, "rejected", request_id, started, error=str(e))
        except Exception as e:  # noqa: BLE001 - tools must never raise into the LLM loop
            status = "unavailable" if _is_unavailable(e) else "error"
            msg = f"{type(e).__name__}: {e}"
            return self._result(spec.name, status, request_id, started, error=msg[:500])
        finally:
            request_id_var.reset(token)
            ex.shutdown(wait=False)

    def execute(self, name: str, raw_args: dict | None, request_id: str) -> ToolResult:
        started = time.perf_counter()
        raw_args = raw_args or {}
        spec = self.tools.get(name)
        if spec is None:
            res = self._result(name, "error", request_id, started, error=f"unknown tool '{name}'")
            self._log(res, None, raw_args)
            return res
        try:
            inp = spec.input_model.model_validate(raw_args)
        except ValidationError as e:
            res = self._result(name, "error", request_id, started, error=f"invalid request: {e}")
            self._log(res, spec, raw_args)
            return res
        args = inp.model_dump()
        reason = self.policy.check(spec.module, name, args)
        if reason:
            res = self._result(name, "rejected", request_id, started, error=reason)
            self._log(res, spec, args)
            return res
        if spec.risk == "mutate":
            ns = args.get("namespace")
            summary = f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})"
            action = self.store.create(name, args, summary, ns)
            res = self._result(
                name,
                "pending_confirmation",
                request_id,
                started,
                data={
                    "action_id": action.action_id,
                    "summary": summary,
                    "expires_at": action.expires_at.isoformat(),
                    "note": "NOT executed. The user must confirm this action.",
                },
            )
            self._log(res, spec, args)
            return res
        res = self._run(spec, inp, request_id, started)
        self._log(res, spec, args)
        return res

    # ---- human decisions (never exposed to the LLM) ----
    def confirm(self, action_id: str, request_id: str = "-") -> ToolResult:
        started = time.perf_counter()
        action, err = self.store.take(action_id, "confirmed")
        if err or action is None:
            return self._result(
                action.tool if action else "unknown", "rejected", request_id, started, error=err
            )
        spec = self.tools[action.tool]
        # re-check policy at execution time, then run the args exactly as stored
        reason = self.policy.check(spec.module, spec.name, action.args)
        if reason:
            return self._result(spec.name, "rejected", request_id, started, error=reason)
        inp = spec.input_model.model_validate(action.args)
        res = self._run(spec, inp, request_id, started)
        action.result = res.model_dump(mode="json")
        self._log(res, spec, action.args)
        return res

    def reject(self, action_id: str) -> bool:
        action, err = self.store.take(action_id, "rejected")
        return err is None and action is not None


registry = Registry()


def tool(
    name: str,
    description: str,
    input_model: type[BaseModel],
    risk: Literal["read", "mutate"] = "read",
    module: str = "",
    timeout_s: float | None = None,
) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        registry.register(ToolSpec(name, description, input_model, fn, risk, module, timeout_s))
        return fn

    return deco
