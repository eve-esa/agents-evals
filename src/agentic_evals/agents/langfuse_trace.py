"""Langfuse trace for LangGraph agents, next to the Weave one (``agents.weave_trace``).

Langfuse's own LangChain handler records agent runs, model turns and tool
calls. One handler serves a whole sample and is pinned to one trace id, so the
orchestrator and every A2A specialist run land in the same trace. Langfuse
logs only when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set
(``LANGFUSE_BASE_URL`` picks the host); otherwise ``langfuse_handler`` returns ``None``.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

ENV_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def langfuse_handler() -> Any | None:
    """A handler for one sample's trace, or ``None`` when Langfuse is not configured."""
    if not all(os.environ.get(key) for key in ENV_KEYS):
        return None
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    if not os.environ.get("LANGFUSE_HOST") and os.environ.get("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
    return CallbackHandler(trace_context={"trace_id": Langfuse.create_trace_id(seed=uuid.uuid4().hex)})


def flush_langfuse() -> None:
    """Send what is buffered; a short run would otherwise end before the export."""
    if all(os.environ.get(key) for key in ENV_KEYS):
        from langfuse import get_client

        get_client().flush()
