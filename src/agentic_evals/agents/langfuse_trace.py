"""Langfuse trace for LangGraph agents, next to the Weave one (``agents.weave_trace``).

Langfuse's own LangChain handler records agent runs, model turns and tool
calls. One handler serves a whole sample and is pinned to one trace id, so the
orchestrator and every A2A specialist run land in the same trace. A specialist's
run nests under the ``ask_*`` tool call that reached it, found through the
``CALLER_RUN_ID`` its run metadata carries (as in ``agents.weave_trace``). Langfuse
logs only when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set
(``LANGFUSE_BASE_URL`` picks the host); otherwise ``langfuse_handler`` returns ``None``.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from agentic_evals.agents.weave_trace import CALLER_RUN_ID

ENV_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def _nesting_handler() -> type:
    from langfuse.langchain import CallbackHandler

    class NestingHandler(CallbackHandler):
        def _take_root_trace_context(self, *, inputs: Any, metadata: dict[str, Any] | None) -> tuple[Any, Any]:
            # Overrides a private hook of the pinned langfuse: where a root run attaches.
            try:
                caller = uuid.UUID(str((metadata or {}).get(CALLER_RUN_ID)))
            except ValueError:
                caller = None
            tool = self._runs.get(caller) if caller else None
            if tool is not None:
                return None, {"trace_id": tool.trace_id, "parent_span_id": tool.id}
            return super()._take_root_trace_context(inputs=inputs, metadata=metadata)

    return NestingHandler


def langfuse_handler() -> Any | None:
    """A handler for one sample's trace, or ``None`` when Langfuse is not configured."""
    if not all(os.environ.get(key) for key in ENV_KEYS):
        return None
    from langfuse import Langfuse
    if not os.environ.get("LANGFUSE_HOST") and os.environ.get("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
    return _nesting_handler()(trace_context={"trace_id": Langfuse.create_trace_id(seed=uuid.uuid4().hex)})


def flush_langfuse() -> None:
    """Send what is buffered; a short run would otherwise end before the export."""
    if all(os.environ.get(key) for key in ENV_KEYS):
        from langfuse import get_client

        get_client().flush()
