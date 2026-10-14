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


def _nesting_handler(attributes: dict[str, Any] | None = None) -> type:
    from langfuse.langchain import CallbackHandler

    class NestingHandler(CallbackHandler):
        def _parse_langfuse_trace_attributes(self, *, metadata: Any, tags: Any) -> dict[str, Any]:
            return {**super()._parse_langfuse_trace_attributes(metadata=metadata, tags=tags), **(attributes or {})}

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


def sample_trace_id(sample_uuid: str) -> str:
    """The Langfuse trace id of a sample; ``agents.langfuse_scores`` derives the same one to attach scores."""
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=sample_uuid)


def langfuse_handler() -> Any | None:
    """A handler for one sample's trace, or ``None`` when Langfuse is not configured.

    Inside an Inspect sample the trace id comes from the sample's uuid, and the
    trace is named ``<task>/<sample id>`` in the session of the eval run, tagged
    with task, model and epoch.
    """
    if not all(os.environ.get(key) for key in ENV_KEYS):
        return None
    from langfuse import Langfuse
    if not os.environ.get("LANGFUSE_HOST") and os.environ.get("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
    from inspect_ai.log._samples import sample_active

    active = sample_active()
    if active is None:
        return _nesting_handler()(trace_context={"trace_id": sample_trace_id(uuid.uuid4().hex)})
    attributes = {
        "session_id": active.run_id,
        "trace_name": f"{active.task}/{active.sample.id}",
        "tags": [active.task, active.model, f"epoch-{active.epoch}"],
    }
    return _nesting_handler(attributes)(trace_context={"trace_id": sample_trace_id(active.sample_uuid)})


def flush_langfuse() -> None:
    """Send what is buffered; a short run would otherwise end before the export."""
    if all(os.environ.get(key) for key in ENV_KEYS):
        from langfuse import get_client

        get_client().flush()
