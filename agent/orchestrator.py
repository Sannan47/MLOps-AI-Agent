"""Orchestrator: manual LLM tool-use loop, evidence collection and answer grounding."""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import agent.tools  # noqa: F401  (registers all tools)
from agent.config import get_settings
from agent.llm import LLMClient, LLMUnavailable
from agent.logging_utils import log_event, redact, request_id_var
from agent.registry import Registry, registry
from agent.schemas import AgentResponse, Fact, ToolCallRecord, ToolResult

MAX_ITERATIONS = 12
TIME_BUDGET_S = 240.0
MAX_RESULT_CHARS = 12000
PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"

# Evidence ids may appear alone, in brackets, or several per bracket: [E-aaaaaa, E-bbbbbb]
CITE_TOKEN = r"E-[0-9a-f]{6}"
CITE_RE = re.compile(r"(?<![0-9A-Za-z])(" + CITE_TOKEN + r")(?![0-9A-Za-z])")
CITE_BRACKET_RE = re.compile(r"\[\s*" + CITE_TOKEN + r"(?:\s*[,;]\s*" + CITE_TOKEN + r")*\s*\]")
ISO_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?"
)
HEX_RE = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")
VERSION_RE = re.compile(r"\b(?:data-)?v?\d+\.\d+\.\d+\b|\bdata-v\d+\.\d+\b|\bv\d+\.\d+\b")
POD_RE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)*-[a-z0-9]{8,10}-[a-z0-9]{5}\b")
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
SECTIONS = {
    "known facts": "facts",
    "uncertainties": "uncertainties",
    "recommended actions": "recommendations",
}


def system_prompt() -> str:
    s = get_settings()
    return (
        PROMPT_PATH.read_text(encoding="utf-8")
        .replace("{namespace}", s.k8s_namespace)
        .replace("{dags}", ", ".join(s.approved_dag_list))
    )


class Evidence:
    """Collected tool results for one request."""

    def __init__(self) -> None:
        self.results: list[ToolResult] = []

    def add(self, r: ToolResult) -> None:
        self.results.append(r)

    @property
    def supporting(self) -> list[ToolResult]:
        """Results whose data may back a claim. A pending action's id and args are real facts."""
        return [r for r in self.results if r.status in ("ok", "pending_confirmation")]

    @property
    def ok_ids(self) -> set[str]:
        return {r.evidence_id for r in self.supporting}

    @property
    def ok_text(self) -> str:
        return " ".join(json.dumps(r.data, default=str) for r in self.supporting).lower()

    @property
    def ok_numbers(self) -> list[float]:
        return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", self.ok_text)]


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def find_unsupported(answer: str, ev: Evidence) -> list[str]:
    """Return tokens in the answer that no ok tool result supports."""
    bad: list[str] = []
    for cid in CITE_RE.findall(answer):
        if cid not in ev.ok_ids:
            bad.append(f"[{cid}]")
    text = CITE_BRACKET_RE.sub(" ", answer)
    text = CITE_RE.sub(" ", text)
    text = ISO_DATE_RE.sub(" ", text)
    ev_text = ev.ok_text
    for rx in (POD_RE, VERSION_RE, HEX_RE):
        for m in rx.finditer(text):
            tok = m.group(0)
            if tok.lower() not in ev_text and tok.lstrip("v").lower() not in ev_text:
                bad.append(tok)
        text = rx.sub(" ", text)
    nums = ev.ok_numbers
    for m in NUM_RE.finditer(text):
        raw = m.group(0)
        pct = raw.endswith("%")
        s = raw.rstrip("%")
        x = float(s)
        if "." not in s and abs(x) <= 12:
            continue  # small counts/list numbers are not treated as claims
        cands = [x / 100] if pct else [x]
        d = _decimals(s)
        ok = any(round(n, d) == c or n == c for n in nums for c in cands)
        if not ok and pct:
            ok = any(round(n * 100, d) == x for n in nums)
        if not ok:
            bad.append(raw)
    return list(dict.fromkeys(bad))


def strip_unsupported(answer: str, bad: list[str]) -> str:
    kept = []
    for line in answer.splitlines():
        if any(b in line for b in bad):
            continue
        kept.append(line)
    return "\n".join(kept)


def parse_sections(answer: str) -> tuple[list[Fact], list[str], list[str]]:
    cur = None
    buckets: dict[str, list[str]] = {"facts": [], "uncertainties": [], "recommendations": []}
    for line in answer.splitlines():
        head = line.strip().strip("*#: ").lower()
        if head in SECTIONS:
            cur = SECTIONS[head]
            continue
        if cur and line.strip():
            buckets[cur].append(line.strip().lstrip("-* ").strip())
    facts = [Fact(statement=s, evidence_ids=CITE_RE.findall(s)) for s in buckets["facts"]]
    return facts, buckets["uncertainties"], buckets["recommendations"]


class Agent:
    def __init__(self, llm: LLMClient, reg: Registry | None = None):
        self.llm = llm
        self.reg = reg or registry

    def _tool_result_block(self, tu_id: str, res: ToolResult) -> dict:
        payload = res.model_dump(mode="json")
        body = json.dumps(redact(payload), default=str)
        if len(body) > MAX_RESULT_CHARS:
            body = body[:MAX_RESULT_CHARS] + '..."truncated"'
        return {
            "type": "tool_result",
            "tool_use_id": tu_id,
            "content": body,
            "is_error": res.status not in ("ok", "pending_confirmation"),
        }

    def run(self, user_message: str, request_id: str) -> AgentResponse:
        token = request_id_var.set(request_id)
        try:
            return self._run(user_message, request_id)
        finally:
            request_id_var.reset(token)

    def _run(self, user_message: str, request_id: str) -> AgentResponse:
        ev = Evidence()
        records: list[ToolCallRecord] = []
        messages: list[dict] = [{"role": "user", "content": user_message}]
        system, tools = system_prompt(), self.reg.anthropic_tool_specs()
        started = time.monotonic()
        answer, partial = "", False
        try:
            for _ in range(MAX_ITERATIONS):
                if time.monotonic() - started > TIME_BUDGET_S:
                    partial = True
                    break
                resp = self.llm.create(system, messages, tools)
                messages.append({"role": "assistant", "content": resp.content})
                if not resp.tool_uses:
                    answer = resp.text
                    break
                blocks = []
                for tu in resp.tool_uses:
                    res = self.reg.execute(tu.name, tu.input, request_id)
                    ev.add(res)
                    records.append(
                        ToolCallRecord(
                            tool=tu.name,
                            args=redact(tu.input),
                            status=res.status,
                            duration_ms=res.duration_ms,
                            evidence_id=res.evidence_id,
                        )
                    )
                    blocks.append(self._tool_result_block(tu.id, res))
                messages.append({"role": "user", "content": blocks})
            else:
                partial = True
            if partial and not answer:
                answer = (
                    "**Known facts**\n- Investigation stopped before completion (iteration or time "
                    "budget reached).\n**Uncertainties**\n- Remaining questions were not answered.\n"
                    "**Recommended actions**\n- Ask a narrower question."
                )
            answer = self._ground(answer, ev, messages, system, tools)
        except LLMUnavailable as e:
            log_event("llm_unavailable", error=str(e))
            answer = (
                f"**Known facts**\n- None: the LLM is unavailable ({e}).\n**Uncertainties**\n- "
                "The request could not be processed.\n**Recommended actions**\n- Retry later."
            )
        answer = self._unavailable_guard(answer, ev, records)
        facts, unc, rec = parse_sections(answer)
        pending = [
            a
            for a in self.reg.store.list()
            if a.action_id
            in {
                r.data.get("action_id")
                for r in ev.results
                if r.status == "pending_confirmation" and isinstance(r.data, dict)
            }
        ]
        resp_obj = AgentResponse(
            request_id=request_id,
            answer=answer,
            facts=facts,
            uncertainties=unc,
            recommendations=rec,
            tool_calls=records,
            pending_actions=pending,
        )
        self._save(user_message, resp_obj, ev)
        return resp_obj

    def _ground(
        self, answer: str, ev: Evidence, messages: list[dict], system: str, tools: list[dict]
    ) -> str:
        bad = find_unsupported(answer, ev)
        if not bad:
            return answer
        log_event("grounding_violation", unsupported=bad, attempt=1)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your answer contains claims not supported by tool results from this conversation: "
                    f"{bad}. Rewrite the answer using only supported facts (cite evidence ids) and put "
                    "anything unverified under Uncertainties. Do not call more tools unless needed."
                ),
            }
        )
        resp = self.llm.create(system, messages, tools)
        revised = resp.text if not resp.tool_uses else answer
        bad2 = find_unsupported(revised, ev)
        if not bad2:
            return revised
        log_event("grounding_violation", unsupported=bad2, attempt=2)
        cleaned = strip_unsupported(revised, bad2)
        note = "- Some claims could not be verified against tool evidence and were removed."
        lines = cleaned.splitlines()
        for i, ln in enumerate(lines):
            if ln.strip().strip("*#: ").lower() == "uncertainties":
                lines.insert(i + 1, note)
                return "\n".join(lines)
        return cleaned + "\n**Uncertainties**\n" + note

    def _unavailable_guard(self, answer: str, ev: Evidence, records: list[ToolCallRecord]) -> str:
        by_module: dict[str, list[str]] = {}
        for r in records:
            spec = self.reg.tools.get(r.tool)
            if spec:
                by_module.setdefault(spec.module, []).append(r.status)
        low = answer.lower()
        notes = []
        for module, statuses in by_module.items():
            if all(s in ("unavailable", "timeout", "error") for s in statuses):
                if (
                    not any(w in low for w in ("could not", "unable", "unavailable", "couldn't"))
                    or module.lower() not in low
                ):
                    notes.append(
                        f"- {module} information could not be retrieved (service unavailable, "
                        "timed out or errored); no conclusions are drawn about it."
                    )
        return answer + ("\n" + "\n".join(notes) if notes else "")

    def _save(self, question: str, resp: AgentResponse, ev: Evidence) -> None:
        if not get_settings().save_transcripts:
            return
        out = Path("evidence/conversations")
        out.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", question.lower())[:40].strip("_")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        doc = {
            "request_id": resp.request_id,
            "question": question,
            "tool_results": [redact(r.model_dump(mode="json")) for r in ev.results],
            "response": resp.model_dump(mode="json"),
        }
        (out / f"{stamp}_{slug}.json").write_text(json.dumps(doc, indent=2, default=str), "utf-8")
