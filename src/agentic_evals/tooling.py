"""The contract between tasks and agents for tool access.

Tasks own their tools; agents stay benchmark-agnostic. A task's setup solver
publishes the MCP servers for the current sample into the sample store under
``MCP_SERVERS_KEY``, as a mapping of server name to a connection config in the
``langchain_mcp_adapters`` format (e.g. ``{"transport": "stdio", "command": ...,
"args": [...], "cwd": ...}``). Agents read it with ``mcp_servers()`` and connect
however they like. Per-sample configs let tasks give every sample its own
output directory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

MCP_SERVERS_KEY = "agentic_evals:mcp_servers"

ServerConfigs = dict[str, dict[str, Any]]


@solver
def publish_mcp_servers(factory: Callable[[TaskState], ServerConfigs]) -> Solver:
    """Setup solver: store ``factory(state)`` as this sample's MCP server configs."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store.set(MCP_SERVERS_KEY, factory(state))
        return state

    return solve


def mcp_servers() -> ServerConfigs:
    """The MCP servers the current sample's task published (empty if none)."""
    return dict(store().get(MCP_SERVERS_KEY, {}))
