"""Our own tool-call trace, written into the Inspect sample store.

With ``agent_bridge()``, Inspect rebuilds ``state.messages`` from the model calls
that pass through the bridge: it sees tool results only as the model saw them,
records no tool events of its own, and follows a single main thread. Scorers
therefore read this trace instead. It is framework-level (a LangChain callback),
so any LangGraph agent, single or multi-agent, records the same way; each entry
carries the name of the agent that made the call.

Trace entry: ``{"step", "agent", "role", "name", "input", "output", "error",
"error_message", "started_at", "ended_at", "run_id", "parent_run_id"}``, plus
``a2a_task_id`` when the run's metadata carries one (a sub-agent serving an A2A
task) and, for a non-domain tool, ``artifact`` when it returned one (an A2A
delegation's ids).

A tool's role is ``ROLE_METADATA_KEY`` in its metadata; an untagged tool is
``"domain"`` (a benchmark tool). The A2A ``ask_*`` tools are ``"peer"``. Traces:

- ``TRACE_KEY``: domain calls only, from every agent. The benchmarks score this
  one, so a call to another agent is never a trajectory step.
- ``RAW_TRACE_KEY``: every tool call of the top-level agent, as it saw them,
  delegations included. Written by multi-agent solvers.
- ``SUBAGENT_TRACES_KEY``: per sub-agent, its conversation and its domain calls.
  Written by multi-agent solvers.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from inspect_ai.util import store
from langchain_core.callbacks import AsyncCallbackHandler

TRACE_KEY = "agentic_evals:trace"
RAW_TRACE_KEY = "agentic_evals:raw_trace"
SUBAGENT_TRACES_KEY = "agentic_evals:subagent_traces"
MAX_OUTPUT_CHARS = 50_000
# Tool metadata key for a tool's role. Untagged tools are "domain".
ROLE_METADATA_KEY = "agentic_evals:role"
DOMAIN = "domain"


def _text(output: Any) -> tuple[str, bool]:
    """Normalise a tool result (ToolMessage, MCP content list, str) to text and an error flag."""
    error = getattr(output, "status", None) == "error"
    content = getattr(output, "content", output)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict):
                parts.append(f"[{part.get('type', 'content')}]")
            else:
                parts.append(str(part))
        text = "\n".join(parts)
    else:
        text = str(content)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + f"... [truncated {len(text) - MAX_OUTPUT_CHARS} chars]"
    return text, error


def _role(metadata: dict[str, Any] | None) -> str:
    return str((metadata or {}).get(ROLE_METADATA_KEY) or DOMAIN)


class TraceRecorder(AsyncCallbackHandler):
    def __init__(
        self,
        agent_name: str,
        entries: list[dict[str, Any]] | None = None,
        roles: tuple[str, ...] | None = (DOMAIN,),
    ) -> None:
        """``entries`` lets several recorders (one per agent) share one trace with global step numbers.

        ``roles`` are the tool roles recorded; ``None`` records every tool call.
        """
        self.agent_name = agent_name
        self.entries: list[dict[str, Any]] = [] if entries is None else entries
        self.roles = roles
        self._open: dict[UUID, dict[str, Any]] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        role = _role(metadata)
        if self.roles is not None and role not in self.roles:
            return
        entry = {
            "step": len(self.entries) + 1,
            "agent": self.agent_name,
            "role": role,
            "name": (serialized or {}).get("name") or kwargs.get("name", ""),
            "input": inputs if inputs is not None else input_str,
            "output": None,
            "error": False,
            "error_message": None,
            "started_at": time.time(),
            "ended_at": None,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
        }
        if (metadata or {}).get("a2a_task_id"):
            entry["a2a_task_id"] = metadata["a2a_task_id"]
        self.entries.append(entry)
        self._open[run_id] = entry

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._open.pop(run_id, None)
        if entry is None:
            return
        entry["output"], entry["error"] = _text(output)
        # An MCP tool's artifact is its structured output again; keep only a delegation's ids.
        artifact = getattr(output, "artifact", None)
        if artifact is not None and entry["role"] != DOMAIN:
            entry["artifact"] = artifact
        entry["ended_at"] = time.time()

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._open.pop(run_id, None)
        if entry is None:
            return
        entry["error"] = True
        entry["error_message"] = f"{type(error).__name__}: {error}"
        entry["ended_at"] = time.time()

    def save(self, key: str = TRACE_KEY) -> None:
        existing = list(store().get(key, []))
        store().set(key, existing + self.entries)


def read_trace(state_store: Any) -> list[dict[str, Any]]:
    """Read the trace from a TaskState's store (``state.store``)."""
    return list(state_store.get(TRACE_KEY, []))


def recorded_trace(state_store: Any) -> list[dict[str, Any]] | None:
    """The trace, or ``None`` when the solver recorded none (not when it recorded no call)."""
    trace = state_store.get(TRACE_KEY)
    return None if trace is None else list(trace)
