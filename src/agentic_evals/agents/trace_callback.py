"""Our own tool-call trace, written into the Inspect sample store.

With ``agent_bridge()``, Inspect rebuilds ``state.messages`` from the model calls
that pass through the bridge: it sees tool results only as the model saw them,
records no tool events of its own, and follows a single main thread. Scorers
therefore read this trace instead. It is framework-level (a LangChain callback),
so any LangGraph agent, single or multi-agent, records the same way; each entry
carries the name of the agent that made the call.

Trace entry: ``{"step", "agent", "name", "input", "output", "error",
"error_message", "started_at", "ended_at", "run_id", "parent_run_id"}``.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from inspect_ai.util import store
from langchain_core.callbacks import AsyncCallbackHandler

TRACE_KEY = "agentic_evals:trace"
MAX_OUTPUT_CHARS = 50_000


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


class TraceRecorder(AsyncCallbackHandler):
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.entries: list[dict[str, Any]] = []
        self._open: dict[UUID, dict[str, Any]] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        entry = {
            "step": len(self.entries) + 1,
            "agent": self.agent_name,
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
        self.entries.append(entry)
        self._open[run_id] = entry

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._open.pop(run_id, None)
        if entry is None:
            return
        entry["output"], entry["error"] = _text(output)
        entry["ended_at"] = time.time()

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._open.pop(run_id, None)
        if entry is None:
            return
        entry["error"] = True
        entry["error_message"] = f"{type(error).__name__}: {error}"
        entry["ended_at"] = time.time()

    def save(self) -> None:
        existing = list(store().get(TRACE_KEY, []))
        store().set(TRACE_KEY, existing + self.entries)


def read_trace(state_store: Any) -> list[dict[str, Any]]:
    """Read the trace from a TaskState's store (``state.store``)."""
    return list(state_store.get(TRACE_KEY, []))
